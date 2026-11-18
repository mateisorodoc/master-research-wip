"""Synchronized multimodal causal pipeline and streaming iterator."""

import numpy as np
from src.data.loader import BASELINE_SAMPLES, DEAPLoader, FS_HZ, MONTAGE_CHANNELS, TOTAL_SAMPLES
from src.features.bvp_pipeline import CausalBVPPipeline
from src.features.eeg_pipeline import CausalEEGPipeline, assemble_eeg_montage, montage_feature_names
from src.features.gsr_pipeline import CausalGSRPipeline


class CausalMultimodalPipeline:
    """Synchronizes causal EEG, GSR, and BVP pipelines onto one discrete delta_t grid."""

    def __init__(self, montage: str = "wearable_6", fs: float = FS_HZ, dt_sec: float = 0.5):
        self.montage = montage
        self.fs = float(fs)
        self.dt_sec = float(dt_sec)
        self.step_samples = int(round(self.dt_sec * self.fs))
        self.eeg = CausalEEGPipeline(montage=montage, fs=self.fs)
        self.gsr = CausalGSRPipeline(fs=self.fs)
        self.bvp = CausalBVPPipeline(fs=self.fs)
        self.feature_names = self.eeg.feature_names + self.gsr.feature_names + self.bvp.feature_names
        self.n_features = len(self.feature_names)
        self.baseline_scl = 0.0
        self.last_omega_bvp = self.bvp.degraded_omega
        self.last_cold_start = True

    def reset(self) -> None:
        self.eeg.reset()
        self.gsr.reset()
        self.bvp.reset()
        self.baseline_scl = 0.0
        self.last_omega_bvp = self.bvp.degraded_omega
        self.last_cold_start = True

    def fit_baseline(self, raw_baseline: np.ndarray) -> None:
        """Fit baseline statistics strictly on pre-stimulus samples (e.g. 0:384)."""
        self.eeg.fit_baseline(raw_baseline)
        self.baseline_scl = self.gsr.fit_baseline(raw_baseline)
        self.bvp.fit_baseline(raw_baseline)

    def step(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Process one (40, L) causal chunk and return unified feature vector y_k in R^D."""
        f_eeg = self.eeg.step(raw_chunk)
        f_gsr = self.gsr.step(raw_chunk)
        f_bvp, omega, cold = self.bvp.step(raw_chunk)
        self.last_omega_bvp = omega
        self.last_cold_start = cold
        return np.concatenate([f_eeg, f_gsr, f_bvp], axis=0)

    def process_trial(self, trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process full (40, 8064) DEAP trial causally -> (features, timestamps, omega_bvp, cold_start)."""
        self.fit_baseline(trial[:, :BASELINE_SAMPLES])
        n_stim = trial.shape[1] - BASELINE_SAMPLES
        n_steps = n_stim // self.step_samples

        feats = np.empty((n_steps, self.n_features), dtype=np.float32)
        timestamps = np.empty(n_steps, dtype=np.float32)
        omega = np.empty(n_steps, dtype=np.float32)
        cold = np.empty(n_steps, dtype=bool)

        for k in range(n_steps):
            s0 = BASELINE_SAMPLES + k * self.step_samples
            s1 = s0 + self.step_samples
            feats[k] = self.step(trial[:, s0:s1]).astype(np.float32)
            timestamps[k] = np.float32((k + 1) * self.dt_sec)
            omega[k] = np.float32(self.last_omega_bvp)
            cold[k] = self.last_cold_start

        return feats, timestamps, omega, cold


def extract_trial_all_montages(
    trial: np.ndarray, dt_sec: float = 0.5, fs: float = float(FS_HZ)
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run one causal 32-ch EEG + peripheral pass and pack all three montages identically."""
    step_samples = int(round(dt_sec * fs))
    eeg32 = CausalEEGPipeline(montage="full_32", fs=fs)
    gsr = CausalGSRPipeline(fs=fs)
    bvp = CausalBVPPipeline(fs=fs)

    base = trial[:, :BASELINE_SAMPLES]
    eeg32.fit_baseline(base)
    gsr.fit_baseline(base)
    bvp.fit_baseline(base)

    n_steps = (TOTAL_SAMPLES - BASELINE_SAMPLES) // step_samples
    montages = ("full_32", "wearable_6", "wearable_4")
    all32_channels = MONTAGE_CHANNELS["full_32"]

    per_montage_feats = {
        m: np.empty((n_steps, len(montage_feature_names(m)) + 11), dtype=np.float32)
        for m in montages
    }
    timestamps = np.empty(n_steps, dtype=np.float32)
    omega = np.empty(n_steps, dtype=np.float32)
    cold = np.empty(n_steps, dtype=bool)

    for k in range(n_steps):
        s0 = BASELINE_SAMPLES + k * step_samples
        chunk = trial[:, s0 : s0 + step_samples]
        log_p32 = eeg32.step_bandpowers(chunk)
        f_periph = np.concatenate([gsr.step(chunk), bvp.step(chunk)[0]], axis=0)
        timestamps[k] = np.float32((k + 1) * dt_sec)
        omega[k] = np.float32(bvp.last_omega_bvp)
        cold[k] = bvp.last_cold_start
        for m in montages:
            f_eeg = assemble_eeg_montage(log_p32, m, all32_channels)
            per_montage_feats[m][k] = np.concatenate([f_eeg, f_periph], axis=0).astype(np.float32)

    return {m: (per_montage_feats[m], timestamps.copy(), omega.copy(), cold.copy()) for m in montages}


class CausalObservationStream:
    """Streaming trial iterator decoupling Phase 1 causal features from Phase 2/3 loops."""

    def __init__(
        self,
        subject_id: int,
        trial_id: int,
        montage: str = "wearable_6",
        dt_sec: float = 0.5,
        dataset_dir: str = "deap-dataset/data_preprocessed_python",
    ):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.montage = montage
        self.dt_sec = float(dt_sec)
        data, labels = DEAPLoader(dataset_dir).load_subject(subject_id)
        self._trial = data[trial_id]
        self.labels = labels[trial_id]
        self.pipeline = CausalMultimodalPipeline(montage=montage, dt_sec=dt_sec)
        self._cursor = BASELINE_SAMPLES

    def reset(self) -> None:
        self.pipeline.fit_baseline(self._trial[:, :BASELINE_SAMPLES])
        self._cursor = BASELINE_SAMPLES

    def step(self, raw_chunk: np.ndarray) -> np.ndarray:
        return self.pipeline.step(raw_chunk)

    def __iter__(self):
        self.reset()
        step_n = self.pipeline.step_samples
        while self._cursor + step_n <= self._trial.shape[1]:
            y_k = self.step(self._trial[:, self._cursor : self._cursor + step_n])
            self._cursor += step_n
            yield y_k
