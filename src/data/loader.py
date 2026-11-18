"""Read-only DEAP dataset loader and Geneva 10-20 channel definitions."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

FS_HZ = 128
BASELINE_SAMPLES = 384   # 3.0 s pre-stimulus fixation @ 128 Hz
TOTAL_SAMPLES = 8064     # 63.0 s total trial length @ 128 Hz
TRIALS_PER_SUBJECT = 40
CHANNELS_PER_TRIAL = 40

# DEAP BioSemi Geneva 10-20 order; MNE's default 'biosemi32' differs at 30/32 slots.
DEAP_GENEVA_1020_CHANNELS = {
    0: "Fp1", 1: "AF3", 2: "F3", 3: "F7", 4: "FC5", 5: "FC1", 6: "C3", 7: "T7",
    8: "CP5", 9: "CP1", 10: "P3", 11: "P7", 12: "PO3", 13: "O1", 14: "Oz", 15: "Pz",
    16: "Fp2", 17: "AF4", 18: "Fz", 19: "F4", 20: "F8", 21: "FC6", 22: "FC2", 23: "Cz",
    24: "C4", 25: "T8", 26: "CP6", 27: "CP2", 28: "P4", 29: "P8", 30: "PO4", 31: "O2",
}
CHANNEL_NAME_TO_INDEX = {name: idx for idx, name in DEAP_GENEVA_1020_CHANNELS.items()}

# Dual notation prevents off-by-one confusion with tEMG (idx 35) and Respiration (idx 37).
GSR_CHANNEL_1BASED = 37
GSR_INDEX = 36
BVP_CHANNEL_1BASED = 39
BVP_INDEX = 38

MONTAGE_CHANNELS = {
    "full_32": list(range(32)),
    "wearable_6": [18, 2, 19, 28, 7, 25],  # Fz, F3, F4, P4, T7, T8
    "wearable_4": [1, 17, 7, 25],          # AF3, AF4, T7, T8
}


def is_valid_sam_label(labels: np.ndarray) -> np.ndarray:
    """Return boolean mask checking positive finite SAM Valence/Arousal ratings."""
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return bool(np.isfinite(arr[0]) and np.isfinite(arr[1]) and (arr[0] > 0.0) and (arr[1] > 0.0))
    return np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 0] > 0.0) & (arr[:, 1] > 0.0)


class DEAPLoader:
    """Strict read-only loader for DEAP preprocessed Python .dat files."""

    def __init__(
        self,
        dataset_dir: str | Path = "deap-dataset/data_preprocessed_python",
        metadata_path: str | Path | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"DEAP directory not found: {self.dataset_dir}")
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.dataset_dir.parent / "Metadata" / "participant_ratings.xls"
        )
        self._metadata_by_subject: dict[int, np.ndarray] | None = None

    def _load_metadata_table(self) -> dict[int, np.ndarray]:
        """Load uninverted [1.00, 9.00] SAM ratings aligned by Participant_id and Experiment_id."""
        if self._metadata_by_subject is not None:
            return self._metadata_by_subject
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"DEAP metadata ratings file not found: {self.metadata_path}")

        df = pd.read_excel(self.metadata_path)
        by_subj: dict[int, np.ndarray] = {}
        for sid in range(1, 33):
            sub = df[df["Participant_id"] == sid].sort_values("Experiment_id")
            if len(sub) != TRIALS_PER_SUBJECT:
                raise ValueError(f"Expected {TRIALS_PER_SUBJECT} metadata rows for subject {sid}, got {len(sub)}")
            by_subj[sid] = sub[["Valence", "Arousal", "Dominance", "Liking"]].to_numpy(dtype=np.float64)
        self._metadata_by_subject = by_subj
        return by_subj

    def subject_path(self, subject_id: int) -> Path:
        if not 1 <= subject_id <= 32:
            raise ValueError(f"subject_id must be in 1..32, got {subject_id}")
        return self.dataset_dir / f"s{subject_id:02d}.dat"

    def load_subject(self, subject_id: int, use_metadata_labels: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Load one subject as non-writeable (data, labels) float64 arrays.

        By default (``use_metadata_labels=True``), restores authoritative ``[1.00, 9.00]``
        SAM ratings from ``participant_ratings.xls`` (sorted by ``Experiment_id``) to
        correct the DEAP Python ``.dat`` Quadrant-1 ``(9.0 - V, 9.0 - A)`` inversion.
        """
        path = self.subject_path(subject_id)
        with path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")

        data = np.asarray(payload["data"], dtype=np.float64)
        raw_labels = np.asarray(payload["labels"], dtype=np.float64)
        if data.shape != (TRIALS_PER_SUBJECT, CHANNELS_PER_TRIAL, TOTAL_SAMPLES):
            raise ValueError(f"{path.name}: unexpected data shape {data.shape}")
        if raw_labels.shape != (TRIALS_PER_SUBJECT, 4):
            raise ValueError(f"{path.name}: unexpected labels shape {raw_labels.shape}")

        if use_metadata_labels:
            meta_labels = self._load_metadata_table()[subject_id].copy()
            # Cross-verify trial alignment against Dominance/Liking in .dat['labels'].
            if not np.allclose(meta_labels[:, 2:], raw_labels[:, 2:], atol=1e-6):
                raise ValueError(f"{path.name}: Dominance/Liking mismatch vs {self.metadata_path.name}")
            labels = meta_labels
        else:
            labels = raw_labels

        # Barrier 2: prevent any downstream code from mutating raw arrays in memory.
        data.flags.writeable = False
        labels.flags.writeable = False
        return data, labels
