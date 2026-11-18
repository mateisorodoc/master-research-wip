"""Causal EEG preprocessing, sliding RFFT bandpowers, and montage ablations."""

import numpy as np
from src.data.loader import DEAP_GENEVA_1020_CHANNELS, MONTAGE_CHANNELS
from src.features.causal_dsp import CausalSOSFilter, impute_causal_nans

EEG_BANDS = (
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0),
    ("gamma", 30.0, 45.0),
)
FULL32_ASYM_PAIRS = (
    (16, 0),   # Fp2 - Fp1
    (17, 1),   # AF4 - AF3
    (19, 2),   # F4 - F3
    (20, 3),   # F8 - F7
    (22, 5),   # FC2 - FC1
    (21, 4),   # FC6 - FC5
    (25, 7),   # T8 - T7
)
ALPHA_IDX = 1
THETA_IDX = 0
BETA_IDX = 2
GAMMA_IDX = 3


def montage_feature_names(montage: str) -> list[str]:
    indices = MONTAGE_CHANNELS[montage]
    names = [
        f"eeg_{b}_{DEAP_GENEVA_1020_CHANNELS[ch]}"
        for ch in indices
        for b, _, _ in EEG_BANDS
    ]
    if montage == "full_32":
        for r, l in FULL32_ASYM_PAIRS:
            names.append(f"eeg_asym_{DEAP_GENEVA_1020_CHANNELS[r]}_{DEAP_GENEVA_1020_CHANNELS[l]}")
    elif montage == "wearable_6":
        names.extend(["eeg_faa_F4_F3", "eeg_theta_beta_ratio_Fz", "eeg_beta_alpha_ratio_P4"])
    elif montage == "wearable_4":
        names.extend(["eeg_asym_AF4_AF3", "eeg_asym_T8_T7"])
    else:
        raise ValueError(f"Unsupported montage: {montage}")
    return names


def assemble_eeg_montage(bandpowers: np.ndarray, montage: str, source_channels: list[int]) -> np.ndarray:
    """Pack per-channel (C, 4) log bandpowers into the target montage vector."""
    idx_map = {ch: i for i, ch in enumerate(source_channels)}
    sel = [idx_map[ch] for ch in MONTAGE_CHANNELS[montage]]
    feats = [bandpowers[sel].reshape(-1)]

    if montage == "full_32":
        asym = [
            bandpowers[idx_map[r], ALPHA_IDX] - bandpowers[idx_map[l], ALPHA_IDX]
            for r, l in FULL32_ASYM_PAIRS
        ]
        feats.append(np.asarray(asym, dtype=np.float64))
    elif montage == "wearable_6":
        faa = bandpowers[idx_map[19], ALPHA_IDX] - bandpowers[idx_map[2], ALPHA_IDX]
        tbr = bandpowers[idx_map[18], THETA_IDX] - bandpowers[idx_map[18], BETA_IDX]
        bar = bandpowers[idx_map[28], BETA_IDX] - bandpowers[idx_map[28], ALPHA_IDX]
        feats.append(np.asarray([faa, tbr, bar], dtype=np.float64))
    elif montage == "wearable_4":
        af = bandpowers[idx_map[17], ALPHA_IDX] - bandpowers[idx_map[1], ALPHA_IDX]
        t = bandpowers[idx_map[25], ALPHA_IDX] - bandpowers[idx_map[7], ALPHA_IDX]
        feats.append(np.asarray([af, t], dtype=np.float64))

    return np.concatenate(feats, axis=0)


class CausalEEGPipeline:
    """Strict causal EEG filter bank + past-only Hamming RFFT feature extractor."""

    def __init__(
        self,
        montage: str = "wearable_6",
        fs: float = 128.0,
        window_sec: float = 2.0,
        order: int = 4,
        bandpass_hz: tuple[float, float] = (4.0, 45.0),
        clamp_sigma: float = 5.0,
        eps: float = 1e-12,
    ):
        if montage not in MONTAGE_CHANNELS:
            raise ValueError(f"Unknown montage {montage}")
        self.montage = montage
        self.channels = list(MONTAGE_CHANNELS[montage])
        self.fs = float(fs)
        self.win_samples = int(round(window_sec * self.fs))
        self.clamp_sigma = float(clamp_sigma)
        self.eps = float(eps)

        self._sos = CausalSOSFilter(order=order, cutoff_hz=bandpass_hz, btype="bandpass", fs=self.fs)
        self._hamming = np.hamming(self.win_samples).astype(np.float64)
        self._psd_scale = 1.0 / (self.fs * float(np.sum(self._hamming ** 2)))
        freqs = np.fft.rfftfreq(self.win_samples, d=1.0 / self.fs)
        self._df = float(freqs[1] - freqs[0])
        self._band_masks = [
            (freqs >= f_lo) & ((freqs <= f_hi) if b == "gamma" else (freqs < f_hi))
            for b, f_lo, f_hi in EEG_BANDS
        ]

        self.feature_names = montage_feature_names(montage)
        self.n_features = len(self.feature_names)
        self.reset()

    def reset(self) -> None:
        self._sos.reset()
        n_ch = len(self.channels)
        self._buf = np.zeros((n_ch, self.win_samples), dtype=np.float64)
        self._filled = 0
        self._last_valid: np.ndarray | None = None
        self._base_mu: np.ndarray | None = None
        self._base_sigma: np.ndarray | None = None
        self._base_log_p: np.ndarray | None = None

    def _bandpowers_from_window(self, win: np.ndarray) -> np.ndarray:
        """Compute ln(P_b + eps) across the 4 bands from shape (C, N_W).

        Note: Computed from positive discrete FFT frequencies; constant periodogram
        scale factors (two-sided vs single-sided) cancel identically under pre-stimulus
        baseline log-subtraction ln(P) - ln(P_base).
        """
        spec = np.fft.rfft(win * self._hamming[None, :], axis=-1)
        psd = (spec.real ** 2 + spec.imag ** 2) * self._psd_scale
        out = np.empty((win.shape[0], len(EEG_BANDS)), dtype=np.float64)
        for i, mask in enumerate(self._band_masks):
            power = np.sum(psd[:, mask], axis=-1) * self._df
            out[:, i] = np.log(power + self.eps)
        return out

    def _clamp_and_filter(self, raw_eeg: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_eeg, dtype=np.float64)
        x, self._last_valid = impute_causal_nans(x, self._last_valid)
        if self._base_sigma is not None and self._base_mu is not None:
            # Clamp extreme impulse bursts around baseline mean without altering steady signals.
            active = self._base_sigma > 1e-6
            if np.any(active):
                bound = self.clamp_sigma * self._base_sigma[active, None]
                x = x.copy()
                x[active] = self._base_mu[active, None] + np.clip(x[active] - self._base_mu[active, None], -bound, bound)
        return self._sos.filter_chunk(x)

    def _append_buffer(self, filtered: np.ndarray) -> None:
        n_new = filtered.shape[1]
        if n_new == 0:
            return
        if n_new >= self.win_samples:
            self._buf[:, :] = filtered[:, -self.win_samples:]
            self._filled = self.win_samples
            return
        self._buf[:, :-n_new] = self._buf[:, n_new:]
        self._buf[:, -n_new:] = filtered
        self._filled = min(self.win_samples, self._filled + n_new)

    def fit_baseline(self, raw_baseline: np.ndarray) -> np.ndarray:
        """Reset state, warm-up causal filter over baseline samples, and lock baseline stats."""
        self.reset()
        sub = np.asarray(raw_baseline[self.channels], dtype=np.float64)
        clean, self._last_valid = impute_causal_nans(sub, None)
        self._base_mu = np.mean(clean, axis=-1)
        self._base_sigma = np.std(clean, axis=-1)

        filtered = self._sos.filter_chunk(clean)
        self._append_buffer(filtered)
        win = self._buf if self._filled >= self.win_samples else self._buf[:, -self._filled:]
        if win.shape[1] < self.win_samples:
            pad = np.zeros((len(self.channels), self.win_samples), dtype=np.float64)
            pad[:, -win.shape[1]:] = win
            win = pad
        self._base_log_p = self._bandpowers_from_window(win)
        return self._base_log_p.copy()

    def step_bandpowers(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Advance causal EEG state by one chunk (40, L) and return (C, 4) log bandpowers."""
        sub = raw_chunk[self.channels]
        filtered = self._clamp_and_filter(sub)
        self._append_buffer(filtered)
        log_p = self._bandpowers_from_window(self._buf)
        if self._base_log_p is not None:
            log_p = log_p - self._base_log_p
        return log_p

    def step(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Advance causal EEG state by one chunk (40, L) and return montage feature vector."""
        log_p = self.step_bandpowers(raw_chunk)
        return assemble_eeg_montage(log_p, self.montage, self.channels)
