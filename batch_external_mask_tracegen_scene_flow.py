#!/usr/bin/env python3
"""Batch TraceGen trajectories using externally supplied first-frame masks."""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from batch_cup_tracegen_scene_flow import (
    TRACEGEN_CV_TO_BASE,
    interp_by_frame_indices,
    load_transform,
    metrics,
    moving_average,
    path_length,
    save_mask_overlay,
    select_tracks_from_mask,
    write_arx_like_hdf5,
)


TASKS = {
    "grasp_cube": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cube_10seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cube_10seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cube/gt_eepose",
        "num": 10,
        "skip": [],
    },
    "grasp_cup": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cup_10seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cup_10seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cup/gt_eepose",
        "num": 10,
        "skip": [],
    },
    "green_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_green_tray_5seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/green_tray_5seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/green_tray/gt_eepose",
        "num": 5,
        "skip": [],
    },
    "yellow_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_yellow_tray_5seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/yellow_tray_5seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/yellow_tray/gt_eepose",
        "num": 5,
        "skip": [],
    },
}


def parse_args():
    p = argparse.ArgumentParser("Run TraceGen pipeline with external first-frame masks.")
    p.add_argument("--work_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--mask_root", type=Path, default=Path("/mnt/workspace/wudong/Gen3R/data/ARX_masks"))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--expert_cut", type=int, default=130)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--min_track_pixels", type=int, default=20)
    p.add_argument("--closed_value", type=float, default=0.0)
    p.add_argument("--open_value", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no_overwrite", action="store_false", dest="overwrite")
    return p.parse_args()


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def prepare_mask(mask_path, rgb0_path, dst_path):
    rgb = Image.open(rgb0_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst_path)
    return np.asarray(mask) > 0, list(mask.size), int(np.count_nonzero(np.asarray(mask) > 0))


def process_one(args, task_name, cfg, idx):
    base = Path(cfg["base"])
    tracegen_dir = Path(cfg["tracegen"]) / f"sequence_{idx}"
    pipeline_npz = tracegen_dir / "pipeline_outputs.npz"
    rgb0 = tracegen_dir / "per_frame_exports/rgb_gt/frame_0000.png"
    expert_hdf5 = Path(cfg["gt"]) / f"episode_{idx}_ee_pose_arx.hdf5"
    ours_hdf5 = base / "ours_scene_flow" / f"{idx}_anchor_scene_flow/pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5"
    out_dir = base / "tracegen_scene_flow" / f"{idx}_tracegen_scene_flow"
    external_mask = args.mask_root / task_name / f"{idx}_first_frame_mask.png"

    for path in [tracegen_dir, pipeline_npz, rgb0, expert_hdf5, ours_hdf5, external_mask, args.work_dir / "T_cam2end-zero.json"]:
        if not path.exists():
            raise FileNotFoundError(path)

    old_summary = load_json(out_dir / "summary.json")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask0 = out_dir / "external_mask_0000.png"
    mask, mask_size, mask_pixels = prepare_mask(external_mask, rgb0, mask0)

    z = np.load(pipeline_npz)
    track_ids, track_counts, all_ids, all_counts = select_tracks_from_mask(z, mask, args.min_track_pixels)
    if track_ids.size == 0:
        raise RuntimeError(f"No TraceGen tracks selected for {task_name} sequence {idx}.")

    sparse_points = z["tracegen_sparse_points"].astype(np.float64)
    frame_indices = z["frame_indices"].astype(np.int64)
    track_points = sparse_points[track_ids]
    delta_cv = track_points - track_points[:, 0:1, :]
    ee_delta_cv = np.nanmedian(delta_cv, axis=0)
    ee_delta_cv[0] = 0.0

    R_cam2zero, t_cam2zero, transform_raw = load_transform(args.work_dir / "T_cam2end-zero.json")
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0])
    legacy_delta = (R_cam2zero @ (opencv_to_opengl @ ee_delta_cv.T)).T
    legacy_delta = moving_average(legacy_delta, args.smooth_window)
    legacy_delta[0] = 0.0

    inferred_delta = (TRACEGEN_CV_TO_BASE @ ee_delta_cv.T).T
    inferred_delta = moving_average(inferred_delta, args.smooth_window)
    inferred_delta[0] = 0.0

    with h5py.File(expert_hdf5, "r") as f:
        expert_pose = f["observations/ee_pose"][:].astype(np.float64)
    export_n = min(int(args.expert_cut), len(expert_pose))
    expert_cmp = expert_pose[:export_n, :3]

    def pose_from_delta(delta):
        delta130 = interp_by_frame_indices(frame_indices, delta, export_n)
        pose = np.tile(expert_pose[0][None, :], (export_n, 1))
        pose[:, :3] = expert_pose[0, :3][None, :] + float(args.scale) * delta130
        return pose

    legacy_pose = pose_from_delta(legacy_delta)
    inferred_pose = pose_from_delta(inferred_delta)
    with h5py.File(ours_hdf5, "r") as f:
        ours_pose = f["observations/ee_pose"][:export_n].astype(np.float64)

    common_attrs = {
        "generated_by": "batch_external_mask_tracegen_scene_flow.py",
        "source_tracegen_dir": str(tracegen_dir),
        "source_pipeline_npz": str(pipeline_npz),
        "source_external_mask0": str(external_mask),
        "mask0_used": str(mask0),
        "selected_track_ids": track_ids.tolist(),
        "selected_track_pixel_counts": track_counts.tolist(),
        "scale": float(args.scale),
        "orientation_source": "constant expert first-frame roll,pitch,yaw placeholder",
    }

    legacy_hdf5 = out_dir / "tracegen_ee_pose_arx_like_130_gripper_closed_then_open.hdf5"
    inferred_hdf5 = out_dir / "tracegen_ee_pose_arx_like_130_inferred_optical_to_ros_gripper_closed_then_open.hdf5"
    write_arx_like_hdf5(
        legacy_hdf5,
        expert_hdf5,
        legacy_pose,
        common_attrs | {"coordinate_chain": "legacy TraceGen OpenCV/RDF -> OpenGL -> T_cam2end-zero rotation -> expert first-frame alignment"},
        args.closed_value,
        args.open_value,
    )
    write_arx_like_hdf5(
        inferred_hdf5,
        expert_hdf5,
        inferred_pose,
        common_attrs
        | {
            "coordinate_chain": "inferred TraceGen optical/RDF to robot ROS/base: base_x=cv_z, base_y=-cv_x, base_z=-cv_y; then expert first-frame alignment",
            "tracegen_cv_to_base_matrix": TRACEGEN_CV_TO_BASE.tolist(),
        },
        args.closed_value,
        args.open_value,
    )

    save_mask_overlay(rgb0, mask, z["tracegen_sparse_uvz"][:, 0, :2], track_ids, out_dir / "frame0000_ee_mask_tracks_overlay.png")
    shutil.copy2(mask0, out_dir / "mask_0000.png")

    ours_cmp = ours_pose[:, :3]
    inferred_cmp = inferred_pose[:, :3]
    legacy_cmp = legacy_pose[:, :3]

    np.savez_compressed(
        out_dir / "tracegen_ee_pipeline_arrays.npz",
        selected_track_ids=track_ids,
        selected_track_pixel_counts=track_counts,
        all_mask_track_ids=all_ids,
        all_mask_track_pixel_counts=all_counts,
        frame_indices=frame_indices,
        mask0=mask.astype(np.uint8),
        ee_delta_cv=ee_delta_cv,
        legacy_delta=legacy_delta,
        inferred_delta=inferred_delta,
        legacy_pose130=legacy_pose,
        inferred_pose130=inferred_pose,
        ours_pose130=ours_pose,
        expert_pose=expert_pose,
        tracegen_cv_to_base=TRACEGEN_CV_TO_BASE,
        opencv_to_opengl=opencv_to_opengl,
        R_cam2zero=R_cam2zero,
        t_cam2zero=t_cam2zero,
        scale=np.asarray(float(args.scale), dtype=np.float64),
    )

    summary = {
        "task": task_name,
        "idx": idx,
        "tracegen_dir": str(tracegen_dir),
        "pipeline_npz": str(pipeline_npz),
        "external_mask0": str(external_mask),
        "mask0_used": str(mask0),
        "mask_size_wh": mask_size,
        "mask_pixels": mask_pixels,
        "expert_hdf5": str(expert_hdf5),
        "ours_hdf5": str(ours_hdf5),
        "out_dir": str(out_dir),
        "scale": float(args.scale),
        "num_tracegen_steps": int(len(frame_indices)),
        "frame_indices": frame_indices.tolist(),
        "num_export_frames": int(export_n),
        "smooth_window": int(args.smooth_window),
        "min_track_pixels": int(args.min_track_pixels),
        "selected_track_ids": track_ids.tolist(),
        "selected_track_pixel_counts": track_counts.tolist(),
        "metrics": {
            "ours_vs_expert": metrics(ours_cmp, expert_cmp),
            "tracegen_legacy_vs_expert": metrics(legacy_cmp, expert_cmp),
            "tracegen_inferred_vs_expert": metrics(inferred_cmp, expert_cmp),
        },
        "path_lengths_first_cut": {
            "expert": path_length(expert_cmp - expert_cmp[0:1]),
            "ours": path_length(ours_cmp - ours_cmp[0:1]),
            "tracegen_legacy": path_length(legacy_cmp - legacy_cmp[0:1]),
            "tracegen_inferred": path_length(inferred_cmp - inferred_cmp[0:1]),
        },
        "transform_raw": transform_raw,
        "tracegen_inferred_transform": {
            "mapping": "base_x=cv_z, base_y=-cv_x, base_z=-cv_y",
            "matrix": TRACEGEN_CV_TO_BASE.tolist(),
        },
        "previous_summary_before_external_mask_overwrite": old_summary,
        "outputs": {
            "legacy_hdf5": str(legacy_hdf5),
            "inferred_hdf5": str(inferred_hdf5),
            "arrays_npz": str(out_dir / "tracegen_ee_pipeline_arrays.npz"),
            "mask_track_overlay": str(out_dir / "frame0000_ee_mask_tracks_overlay.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "tracegen_inferred_transform_summary.json").write_text(
        json.dumps(summary["tracegen_inferred_transform"] | {"metrics": summary["metrics"]["tracegen_inferred_vs_expert"]}, indent=2)
    )
    (out_dir / "README.md").write_text(
        f"# {idx}_tracegen_scene_flow\n\n"
        f"TraceGen converted with the external first-frame mask for `{task_name}` sequence `{idx}`. "
        "The inferred HDF5 uses fixed sparse tracks selected by the external mask and "
        "`base_x=cv_z, base_y=-cv_x, base_z=-cv_y`, then aligns frame 0 to the expert trajectory.\n"
    )
    return summary


def main():
    args = parse_args()
    all_results = {}
    for task_name in args.tasks:
        cfg = TASKS[task_name]
        results = []
        for idx in range(int(cfg["num"])):
            if idx in set(cfg.get("skip", [])):
                print(f"\n=== Skipping TraceGen {task_name} sequence {idx} by config ===", flush=True)
                continue
            print(f"\n=== Processing TraceGen {task_name} sequence {idx} with external mask ===", flush=True)
            results.append(process_one(args, task_name, cfg, idx))
        out_root = Path(cfg["base"]) / "tracegen_scene_flow"
        batch = {
            "task": task_name,
            "tracegen_root": cfg["tracegen"],
            "out_root": str(out_root),
            "mask_root": str(args.mask_root / task_name),
            "scale": float(args.scale),
            "expert_cut": int(args.expert_cut),
            "skipped_indices": cfg.get("skip", []),
            "results": results,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "external_mask_tracegen_batch_summary.json").write_text(json.dumps(batch, indent=2))
        all_results[task_name] = batch
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
