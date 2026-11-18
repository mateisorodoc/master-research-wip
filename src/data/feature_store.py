"""Windows-safe staged HDF5 and flat NPZ storage for Phase 1 extracted features."""

from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np

from src.data.loader import FS_HZ, is_valid_sam_label
from src.features.bvp_pipeline import BVP_FEATURE_NAMES
from src.features.eeg_pipeline import montage_feature_names
from src.features.gsr_pipeline import GSR_FEATURE_NAMES


def all_feature_names(montage: str) -> list[str]:
    return montage_feature_names(montage) + GSR_FEATURE_NAMES + BVP_FEATURE_NAMES


def write_subject_staging_h5(
    staging_path: Path,
    subject_id: int,
    montage: str,
    trial_results: list[dict],
) -> None:
    """Write one subject's 40 trials to a private HDF5 staging file (avoids Windows file locks)."""
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(staging_path, "w") as hf:
        grp = hf.create_group(f"subject_{subject_id:02d}")
        for idx, res in enumerate(trial_results):
            tgrp = grp.create_group(f"trial_{idx + 1:02d}")
            feats = res["features"]
            tgrp.create_dataset("features", data=feats, dtype="float32", chunks=feats.shape)
            tgrp.create_dataset("timestamps", data=res["timestamps"], dtype="float32")
            tgrp.create_dataset("labels", data=res["labels"], dtype="float32")
            tgrp.create_dataset("omega_bvp_series", data=res["omega_bvp"], dtype="float32")
            tgrp.attrs["trial_number"] = idx + 1
            tgrp.attrs["valid_label"] = bool(is_valid_sam_label(res["labels"]))
            tgrp.attrs["baseline_scl"] = float(res["baseline_scl"])
            tgrp.attrs["quality_omega_bvp"] = float(res["omega_bvp"][-1])
            tgrp.attrs["cold_start"] = bool(res["cold_start"][-1])


def merge_staging_to_outputs(
    staging_paths: list[Path],
    out_h5: Path,
    out_npz: Path,
    montage: str,
    dt_sec: float = 0.5,
) -> dict[str, int]:
    """Merge per-subject staging files single-threaded into master HDF5 and flat NPZ."""
    if not staging_paths:
        raise ValueError("No staging paths provided for merge.")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    feat_names = all_feature_names(montage)

    x_blocks, y_blocks, valid_blocks = [], [], []
    sid_blocks, tid_blocks, ts_blocks, omega_blocks = [], [], [], []

    with h5py.File(out_h5, "w") as master:
        master.attrs["dataset_name"] = "DEAP_Phase1_Causal_Multimodal_Features"
        master.attrs["montage"] = montage
        master.attrs["sampling_rate_raw_hz"] = FS_HZ
        master.attrs["control_step_dt_sec"] = float(dt_sec)
        master.attrs["window_eeg_sec"] = 2.0
        master.attrs["window_hrv_sec"] = 30.0
        master.attrs["raw_dataset_sha256_manifest"] = "docs/deap_sha256_manifest.json"
        master.attrs["extraction_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        master.attrs["feature_names"] = np.asarray(feat_names, dtype=object)

        for spath in sorted(staging_paths):
            with h5py.File(spath, "r") as sf:
                for subj_key in sf.keys():
                    sf.copy(subj_key, master)
                    sid = int(subj_key.split("_")[1])
                    sgrp = sf[subj_key]
                    for tkey in sorted(sgrp.keys()):
                        tgrp = sgrp[tkey]
                        tid = int(tgrp.attrs["trial_number"])
                        feats = tgrp["features"][:]
                        k_len = feats.shape[0]
                        lbl = tgrp["labels"][:]
                        vflag = bool(tgrp.attrs["valid_label"])

                        x_blocks.append(feats)
                        y_blocks.append(np.broadcast_to(lbl, (k_len, 4)).copy())
                        valid_blocks.append(np.full(k_len, vflag, dtype=bool))
                        sid_blocks.append(np.full(k_len, sid, dtype=np.int32))
                        tid_blocks.append(np.full(k_len, tid, dtype=np.int32))
                        ts_blocks.append(tgrp["timestamps"][:])
                        omega_blocks.append(tgrp["omega_bvp_series"][:])

    X = np.concatenate(x_blocks, axis=0).astype(np.float32)
    y = np.concatenate(y_blocks, axis=0).astype(np.float32)
    valid_mask = np.concatenate(valid_blocks, axis=0)
    subject_ids = np.concatenate(sid_blocks, axis=0)
    trial_ids = np.concatenate(tid_blocks, axis=0)
    timestamps = np.concatenate(ts_blocks, axis=0).astype(np.float32)
    omega_bvp = np.concatenate(omega_blocks, axis=0).astype(np.float32)

    np.savez(
        out_npz,
        X=X,
        y=y,
        valid_mask=valid_mask,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        timestamps=timestamps,
        omega_bvp=omega_bvp,
        feature_names=np.asarray(feat_names),
    )

    for spath in staging_paths:
        spath.unlink(missing_ok=True)

    return {"windows": int(X.shape[0]), "features": int(X.shape[1]), "valid_windows": int(np.sum(valid_mask))}
