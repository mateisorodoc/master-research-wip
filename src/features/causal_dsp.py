"""Causal SOS IIR filtering, step-response state caching, and NaN imputation."""

import warnings
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz


class SensorDropoutWarning(UserWarning):
    """Emitted when NaN values trigger causal sample-and-hold imputation."""


def impute_causal_nans(
    chunk: np.ndarray, last_valid: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill NaNs strictly forward along the time axis without lookahead."""
    arr = np.asarray(chunk, dtype=np.float64)
    was_1d = arr.ndim == 1
    work = arr[None, :] if was_1d else arr
    c_dim, t_dim = work.shape

    prev = np.zeros(c_dim, dtype=np.float64) if last_valid is None else np.asarray(last_valid, dtype=np.float64).copy()
    prev = np.where(np.isfinite(prev), prev, 0.0)

    has_nan = not np.all(np.isfinite(work))
    if not has_nan:
        new_last = work[:, -1].copy() if t_dim > 0 else prev
        return (work[0] if was_1d else work.copy()), (new_last[0:1] if was_1d else new_last)

    warnings.warn("Non-finite sensor sample(s) imputed causally via sample-and-hold.", SensorDropoutWarning, stacklevel=2)
    out = work.copy()
    for ch in range(c_dim):
        cur = prev[ch]
        for t in range(t_dim):
            val = out[ch, t]
            if np.isfinite(val):
                cur = val
            else:
                out[ch, t] = cur
        prev[ch] = cur

    return (out[0] if was_1d else out), (prev[0:1] if was_1d else prev)


class CausalSOSFilter:
    """Forward-only Second-Order Sections IIR filter with cached state zi."""

    def __init__(self, order: int, cutoff_hz: float | tuple[float, float], btype: str, fs: float = 128.0):
        self.order = int(order)
        self.cutoff_hz = cutoff_hz
        self.btype = btype
        self.fs = float(fs)
        self.sos = butter(self.order, self.cutoff_hz, btype=self.btype, fs=self.fs, output="sos").astype(np.float64)
        self._zi_unit = sosfilt_zi(self.sos).astype(np.float64)  # shape (n_sections, 2)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def _init_zi(self, first_sample: np.ndarray) -> np.ndarray:
        if first_sample.ndim == 0:
            return self._zi_unit * float(first_sample)
        return self._zi_unit[:, None, :] * first_sample[None, :, None]

    def filter_chunk(self, x_chunk: np.ndarray) -> np.ndarray:
        """Filter a 1D (L,) or 2D (C, L) float64 chunk strictly forward in time."""
        x = np.asarray(x_chunk, dtype=np.float64)
        if x.shape[-1] == 0:
            return x.copy()
        if self._zi is None:
            x0 = x[0] if x.ndim == 1 else x[:, 0]
            self._zi = self._init_zi(x0)
        y, self._zi = sosfilt(self.sos, x, axis=-1, zi=self._zi)
        return y

    def group_delay_samples(self, f_low: float, f_high: float, n_pts: int = 4096) -> tuple[float, float]:
        """Return (mean_delay_samples, peak_delay_samples) within [f_low, f_high] Hz."""
        w, h = sosfreqz(self.sos, worN=n_pts, fs=self.fs)
        phase = np.unwrap(np.angle(h))
        dw = 2.0 * np.pi * (w[1] - w[0]) / self.fs
        tau = -np.gradient(phase, dw)
        mask = (w >= f_low) & (w <= f_high)
        band_tau = tau[mask]
        return float(np.mean(band_tau)), float(np.max(band_tau))


class CausalFilterBank:
    """Collection of named CausalSOSFilter instances sharing identical sampling rate."""

    def __init__(self, bands: dict[str, tuple[float, float]], order: int = 4, fs: float = 128.0):
        self.fs = float(fs)
        self.filters = {
            name: CausalSOSFilter(order=order, cutoff_hz=bounds, btype="bandpass", fs=self.fs)
            for name, bounds in bands.items()
        }

    def reset(self) -> None:
        for flt in self.filters.values():
            flt.reset()

    def filter_chunk(self, x_chunk: np.ndarray) -> dict[str, np.ndarray]:
        return {name: flt.filter_chunk(x_chunk) for name, flt in self.filters.items()}
