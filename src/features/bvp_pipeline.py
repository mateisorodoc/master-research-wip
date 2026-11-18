"""Causal thumb BVP pipeline (Channel 39 / Array Index 38) with HRV fallbacks."""

from collections import deque
import math
import numpy as np
from scipy.signal import lombscargle
from src.data.loader import BVP_INDEX
from src.features.causal_dsp import CausalSOSFilter, impute_causal_nans

BVP_FEATURE_NAMES = [
    "bvp_hr_mean",
    "bvp_hrv_sdnn",
    "bvp_hrv_rmssd",
    "bvp_hrv_lf",
    "bvp_hrv_hf",
    "bvp_hrv_lf_hf",
]
_DEFAULT_FALLBACK = (70.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class CausalBVPPipeline:
    """Strictly causal BVP bandpass + 1-sample-delayed systolic peak HRV extractor."""

    def __init__(
        self,
        fs: float = 128.0,
        window_sec: float = 30.0,
        peak_win_sec: float = 2.0,
        bandpass_hz: tuple[float, float] = (0.5, 5.0),
        order: int = 2,
        threshold_gamma: float = 0.60,
        refractory_sec: float = 0.30,
        min_beats_lfhf: int = 10,
        min_lfhf_sec: float = 30.0,
        degraded_omega: float = 0.1,
        clamp_sigma: float = 5.0,
        eps: float = 1e-12,
    ):
        self.fs = float(fs)
        self.win_samples = int(round(window_sec * self.fs))
        self.peak_win_samples = int(round(peak_win_sec * self.fs))
        self.threshold_gamma = float(threshold_gamma)
        # Ceiling guarantees refractory interval >= 0.30 s (HR < 200 bpm).
        self.refract_samples = int(math.ceil(refractory_sec * self.fs))
        self.min_beats_lfhf = int(min_beats_lfhf)
        self.min_lfhf_samples = int(round(min_lfhf_sec * self.fs))
        self.degraded_omega = float(degraded_omega)
        self.clamp_sigma = float(clamp_sigma)
        self.eps = float(eps)
        self.feature_names = list(BVP_FEATURE_NAMES)

        self._flt = CausalSOSFilter(order=order, cutoff_hz=bandpass_hz, btype="bandpass", fs=self.fs)
        self._lf_w = 2.0 * np.pi * np.linspace(0.04, 0.15, 64)
        self._hf_w = 2.0 * np.pi * np.linspace(0.15, 0.40, 64)
        self.reset()

    def reset(self) -> None:
        self._flt.reset()
        self._peak_buf: deque[float] = deque(maxlen=self.peak_win_samples)
        self._beat_samples: deque[int] = deque()
        self._last_valid: np.ndarray | None = None
        self._base_mu: float | None = None
        self._base_sigma: float | None = None
        self._fallback = np.asarray(_DEFAULT_FALLBACK, dtype=np.float64)
        self._sample_idx = 0
        self._stim_start_idx = 0
        self._prev2 = 0.0
        self._prev1 = 0.0
        self._last_peak_idx = -10**9
        self.last_omega_bvp = self.degraded_omega
        self.last_cold_start = True

    def _preprocess(self, x_raw: np.ndarray, clamp: bool = True) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        x, self._last_valid = impute_causal_nans(x, self._last_valid)
        if clamp and self._base_sigma is not None and self._base_mu is not None and self._base_sigma > 1e-6:
            bound = self.clamp_sigma * self._base_sigma
            x = self._base_mu + np.clip(x - self._base_mu, -bound, bound)
        return x

    def _ingest_filtered(self, x_clean: np.ndarray) -> None:
        y = self._flt.filter_chunk(x_clean)
        for cur in y:
            if self._sample_idx >= 2 and self._peak_buf:
                thresh = max(self.threshold_gamma * max(self._peak_buf), 1e-6)
                if (
                    self._prev1 > self._prev2
                    and self._prev1 >= cur
                    and self._prev1 >= thresh
                    and (self._sample_idx - 1 - self._last_peak_idx) >= self.refract_samples
                ):
                    peak_i = self._sample_idx - 1
                    self._last_peak_idx = peak_i
                    self._beat_samples.append(peak_i)

            self._peak_buf.append(float(cur))
            self._prev2 = self._prev1
            self._prev1 = float(cur)
            self._sample_idx += 1

        cutoff_idx = self._sample_idx - self.win_samples
        while self._beat_samples and self._beat_samples[0] < cutoff_idx:
            self._beat_samples.popleft()

    def _compute_metrics(self) -> tuple[np.ndarray, float, bool]:
        beats = np.fromiter(self._beat_samples, dtype=np.float64) / self.fs
        m = beats.size
        cold_start = ((self._sample_idx - self._stim_start_idx) < self.min_lfhf_samples) or (m < self.min_beats_lfhf)

        if m < 2:
            return self._fallback.copy(), self.degraded_omega, cold_start

        ibis = np.diff(beats)
        hr_mean = float(np.mean(60.0 / ibis))
        sdnn = float(np.std(ibis, ddof=1)) if ibis.size >= 2 else float(self._fallback[1])
        rmssd = float(np.sqrt(np.mean(np.diff(ibis) ** 2))) if ibis.size >= 2 else float(self._fallback[2])

        if cold_start:
            return np.asarray(
                [hr_mean, sdnn, rmssd, self._fallback[3], self._fallback[4], self._fallback[5]],
                dtype=np.float64,
            ), 1.0, True

        t_ibi = 0.5 * (beats[:-1] + beats[1:])
        centered = ibis - float(np.mean(ibis))
        # Degenerate constant-IBI trains have zero variance; return zero spectral power safely.
        if float(np.max(np.abs(centered))) < 1e-12:
            lf, hf, lfhf = 0.0, 0.0, 0.0
        else:
            p_lf = lombscargle(t_ibi, centered, self._lf_w)
            p_hf = lombscargle(t_ibi, centered, self._hf_w)
            trapz = getattr(np, "trapezoid", np.trapz)
            lf = float(trapz(p_lf, self._lf_w / (2.0 * np.pi)))
            hf = float(trapz(p_hf, self._hf_w / (2.0 * np.pi)))
            lfhf = float(lf / (hf + self.eps))

        return np.asarray([hr_mean, sdnn, rmssd, lf, hf, lfhf], dtype=np.float64), 1.0, False

    def fit_baseline(self, raw_baseline: np.ndarray) -> np.ndarray:
        """Warm up causal filter and compute baseline HR/HRV fallbacks over 0:384."""
        self.reset()
        x_raw = raw_baseline[BVP_INDEX] if raw_baseline.ndim == 2 else raw_baseline
        x = self._preprocess(x_raw, clamp=False)
        self._base_mu = float(np.mean(x))
        self._base_sigma = float(np.std(x))
        self._ingest_filtered(x)
        self._stim_start_idx = self._sample_idx
        metrics, _, _ = self._compute_metrics()
        self._fallback = metrics.copy()
        return self._fallback.copy()

    def step(self, raw_chunk: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Consume raw chunk (40, L) or (L,) -> (6 features, omega_bvp, cold_start)."""
        x_raw = raw_chunk[BVP_INDEX] if raw_chunk.ndim == 2 else raw_chunk
        x = self._preprocess(x_raw, clamp=True)
        self._ingest_filtered(x)
        metrics, omega, cold_start = self._compute_metrics()
        self.last_omega_bvp = omega
        self.last_cold_start = cold_start
        return metrics, omega, cold_start
