"""Causal finger GSR pipeline (Channel 37 / Array Index 36)."""

from collections import deque
import numpy as np
from src.data.loader import GSR_INDEX
from src.features.causal_dsp import CausalSOSFilter, impute_causal_nans

GSR_FEATURE_NAMES = [
    "gsr_scl_mean",
    "gsr_scl_slope",
    "gsr_scr_power",
    "gsr_scr_rate",
    "gsr_scr_mean_amp",
]


class CausalGSRPipeline:
    """Strictly causal tonic SCL lowpass + dedicated phasic SCR bandpass feature extractor."""

    def __init__(
        self,
        fs: float = 128.0,
        window_sec: float = 30.0,
        scl_lowpass_hz: float = 0.05,
        scr_bandpass_hz: tuple[float, float] = (0.05, 1.5),
        order: int = 2,
        scr_gamma: float = 0.20,
        refractory_sec: float = 1.0,
        clamp_sigma: float = 5.0,
    ):
        self.fs = float(fs)
        self.win_samples = int(round(window_sec * self.fs))
        self.scr_gamma = float(scr_gamma)
        self.refract_samples = int(round(refractory_sec * self.fs))
        self.clamp_sigma = float(clamp_sigma)
        self.feature_names = list(GSR_FEATURE_NAMES)

        self._tonic_flt = CausalSOSFilter(order=order, cutoff_hz=scl_lowpass_hz, btype="lowpass", fs=self.fs)
        self._phasic_flt = CausalSOSFilter(order=order, cutoff_hz=scr_bandpass_hz, btype="bandpass", fs=self.fs)
        self.reset()

    def reset(self) -> None:
        self._tonic_flt.reset()
        self._phasic_flt.reset()
        self._tonic_hist: deque[float] = deque(maxlen=self.win_samples)
        self._phasic_hist: deque[float] = deque(maxlen=self.win_samples)
        self._peaks: deque[tuple[int, float]] = deque()
        self._last_valid: np.ndarray | None = None
        self._base_mu: float | None = None
        self._base_sigma: float | None = None
        self._scr_thresh = 1e-6
        self._sample_idx = 0
        self._prev2 = 0.0
        self._prev1 = 0.0
        self._last_peak_idx = -10**9

    def _preprocess(self, x_raw: np.ndarray, clamp: bool = True) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        x, self._last_valid = impute_causal_nans(x, self._last_valid)
        if clamp and self._base_sigma is not None and self._base_mu is not None and self._base_sigma > 1e-6:
            bound = self.clamp_sigma * self._base_sigma
            x = self._base_mu + np.clip(x - self._base_mu, -bound, bound)
        return x

    def _ingest_filtered(self, x_clean: np.ndarray) -> None:
        tonic = self._tonic_flt.filter_chunk(x_clean)
        phasic = self._phasic_flt.filter_chunk(x_clean)
        self._tonic_hist.extend(tonic.tolist())
        self._phasic_hist.extend(phasic.tolist())

        # 1-sample delayed local peak detector: checks prev1 once sample k+1 arrives.
        for cur in phasic:
            if self._sample_idx >= 2:
                if (
                    self._prev1 > self._prev2
                    and self._prev1 >= cur
                    and self._prev1 >= self._scr_thresh
                    and (self._sample_idx - 1 - self._last_peak_idx) >= self.refract_samples
                ):
                    peak_i = self._sample_idx - 1
                    self._last_peak_idx = peak_i
                    self._peaks.append((peak_i, float(self._prev1)))
            self._prev2 = self._prev1
            self._prev1 = float(cur)
            self._sample_idx += 1

        cutoff_idx = self._sample_idx - self.win_samples
        while self._peaks and self._peaks[0][0] < cutoff_idx:
            self._peaks.popleft()

    def fit_baseline(self, raw_baseline: np.ndarray) -> float:
        """Fit baseline ADC statistics from 3.0 s fixation without touching active trials."""
        self.reset()
        x_raw = raw_baseline[GSR_INDEX] if raw_baseline.ndim == 2 else raw_baseline
        x = self._preprocess(x_raw, clamp=False)
        self._base_mu = float(np.mean(x))
        self._base_sigma = float(np.std(x))
        self._scr_thresh = max(self.scr_gamma * self._base_sigma, 1e-6)
        self._ingest_filtered(x)
        return self._base_mu

    def step(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Consume one raw chunk (40, L) or (L,) and return 5 causal GSR features."""
        x_raw = raw_chunk[GSR_INDEX] if raw_chunk.ndim == 2 else raw_chunk
        n_chunk = x_raw.shape[-1]
        x = self._preprocess(x_raw, clamp=True)
        self._ingest_filtered(x)

        n_avail = len(self._tonic_hist)
        if n_avail == 0:
            return np.zeros(5, dtype=np.float64)

        tonic_arr = np.fromiter(self._tonic_hist, dtype=np.float64, count=n_avail)
        phasic_arr = np.fromiter(self._phasic_hist, dtype=np.float64, count=n_avail)
        scl_mean = float(np.mean(tonic_arr))
        diff_lag = min(n_avail - 1, max(1, n_chunk))
        scl_slope = float((tonic_arr[-1] - tonic_arr[-1 - diff_lag]) / (diff_lag / self.fs)) if n_avail > 1 else 0.0
        scr_power = float(np.mean(phasic_arr ** 2))

        win_sec = n_avail / self.fs
        n_peaks = len(self._peaks)
        scr_rate = float(n_peaks * (60.0 / win_sec)) if n_peaks > 0 else 0.0
        scr_amp = float(np.mean([amp for _, amp in self._peaks])) if n_peaks > 0 else 0.0
        return np.asarray([scl_mean, scl_slope, scr_power, scr_rate, scr_amp], dtype=np.float64)
