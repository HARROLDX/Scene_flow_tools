#!/usr/bin/env python3
"""Build an ARX-like EE trajectory HDF5 from VAE scene flow."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_args():
    root = Path("/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_geometry_outputs_v2")
    work_dir = root / "grasp_cube_test_scene_flow"
    sample_dir = root / "sample_01_grasp_cube_video_0"
    p = argparse.ArgumentParser("Scene-flow EE trajectory pipeline for the grasp-cube sample.")
    p.add_argument("--sample_dir", type=Path, default=sample_dir)
    p.add_argument("--work_dir", type=Path, default=work_dir)
    p.add_argument("--pipeline_npz", type=Path, default=None)
    p.add_argument("--rgb_dir", type=Path, default=None)
    p.add_argument("--mask0", type=Path, default=None)
    p.add_argument("--flow_key", type=str, default="vae_scene_flow")
    p.add_argument("--point_key", type=str, default="vae_point_maps")
    p.add_argument("--expert_hdf5", type=Path, default=work_dir / "episode_0_ee_pose_arx.hdf5")
    p.add_argument("--cam2zero_json", type=Path, default=work_dir / "T_cam2end-zero.json")
    p.add_argument("--out_dir", type=Path, default=work_dir / "final_scene_flow_arx_pipeline_scale0p6")
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--expert_cut", type=int, default=130)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--agg", choices=["median", "trimmed_mean"], default="median")
    p.add_argument("--trim_frac", type=float, default=0.20)
    p.add_argument("--run_sam_if_missing", action="store_true", default=True)
    p.add_argument("--skip_sam_if_missing", action="store_false", dest="run_sam_if_missing")
    return p.parse_args()


def quat_xyzw_to_rot(q):
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        raise ValueError("Zero quaternion in transform JSON.")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def moving_average(x, window):
    window = int(window)
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([xp[i : i + window].mean(axis=0) for i in range(len(x))], axis=0)


def robust_aggregate(delta_pts, mode, trim_frac):
    if mode == "median":
        return np.median(delta_pts, axis=1)
    out = []
    for pts in delta_pts:
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        keep_n = max(8, int(round((1.0 - trim_frac) * len(dist))))
        keep = np.argsort(dist)[:keep_n]
        out.append(pts[keep].mean(axis=0))
    return np.asarray(out, dtype=np.float64)


def path_length(x):
    return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())


def metrics(pred_xyz, expert_xyz):
    err_vec = pred_xyz - expert_xyz
    err = np.linalg.norm(err_vec, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "final": float(err[-1]),
        "axis_rmse_xyz": np.sqrt(np.mean(err_vec * err_vec, axis=0)).tolist(),
    }


def ensure_mask(args, mask0, rgb_dir):
    if mask0.exists():
        return
    if not args.run_sam_if_missing:
        raise FileNotFoundError(f"mask0 not found: {mask0}")
    sam_script = args.work_dir / "segment_ee_with_sam2.py"
    if not sam_script.exists():
        raise FileNotFoundError(f"SAM2 helper script not found: {sam_script}")
    subprocess.run(
        [
            "python3",
            str(sam_script),
            "--rgb_dir",
            str(rgb_dir),
            "--out_dir",
            str(args.work_dir / "sam2_ee_outputs"),
        ],
        check=True,
    )
    if not mask0.exists():
        raise FileNotFoundError(f"SAM2 completed, but mask0 is still missing: {mask0}")


def load_transform(path):
    data = json.loads(path.read_text())
    R = quat_xyzw_to_rot(data["orientation"])
    t = np.asarray(data["position"], dtype=np.float64)
    return R, t, data


def interpolate_pose(sampled_indices, pose49, total_frames):
    timeline = np.arange(int(total_frames), dtype=np.float64)
    sampled = sampled_indices.astype(np.float64)
    out = np.zeros((len(timeline), pose49.shape[1]), dtype=np.float64)
    for axis in range(pose49.shape[1]):
        out[:, axis] = np.interp(timeline, sampled, pose49[:, axis])
    return out


def save_overlay(rgb0, mask, out_path):
    rgb = np.asarray(Image.open(rgb0).convert("RGB"), dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0.0, 255.0, 80.0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def write_compare_csv(path, expert_xyz, pred_xyz):
    err = np.linalg.norm(pred_xyz - expert_xyz, axis=1)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "expert_x",
                "expert_y",
                "expert_z",
                "pred_x",
                "pred_y",
                "pred_z",
                "err",
            ]
        )
        for i in range(len(expert_xyz)):
            w.writerow([i] + expert_xyz[i].tolist() + pred_xyz[i].tolist() + [float(err[i])])


def copy_attrs(src_h5, dst_h5, extra):
    for k, v in src_h5.attrs.items():
        dst_h5.attrs[k] = v
    for k, v in extra.items():
        if isinstance(v, (dict, list, tuple)):
            dst_h5.attrs[k] = json.dumps(v)
        else:
            dst_h5.attrs[k] = v


def write_arx_like_hdf5(path, template_hdf5, ee_pose, attrs):
    n = int(len(ee_pose))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(template_hdf5, "r") as src, h5py.File(path, "w") as dst:
        copy_attrs(src, dst, attrs)
        obs = dst.create_group("observations")
        ee_pose32 = ee_pose.astype(np.float32)
        obs.create_dataset("ee_pose", data=ee_pose32)
        dst.create_dataset("action_ee_pose", data=ee_pose32)

        dpos = np.diff(ee_pose32[:, :3], axis=0, prepend=ee_pose32[0:1, :3])
        action_delta = np.zeros((n, 5), dtype=np.float32)
        action_delta[:, :3] = dpos
        dst.create_dataset("action_delta_ee", data=action_delta)

        gripper0 = float(src["observations/gripper"][0]) if "observations/gripper" in src else 0.0
        obs.create_dataset("gripper", data=np.full((n,), gripper0, dtype=np.float32))
        dst.create_dataset("action_gripper", data=np.full((n,), gripper0, dtype=np.float32))

        qpos0 = src["observations/qpos"][0].astype(np.float32) if "observations/qpos" in src else np.zeros((7,), np.float32)
        qvel0 = src["observations/qvel"][0].astype(np.float32) if "observations/qvel" in src else np.zeros((7,), np.float32)
        effort0 = src["observations/effort"][0].astype(np.float32) if "observations/effort" in src else np.zeros((7,), np.float32)
        action0 = src["action"][0].astype(np.float32) if "action" in src else np.zeros((7,), np.float32)
        obs.create_dataset("qpos", data=np.tile(qpos0[None, :], (n, 1)))
        obs.create_dataset("qvel", data=np.tile(qvel0[None, :], (n, 1)))
        obs.create_dataset("effort", data=np.tile(effort0[None, :], (n, 1)))
        dst.create_dataset("action", data=np.tile(action0[None, :], (n, 1)))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_npz = args.pipeline_npz or (args.sample_dir / "pipeline_outputs.npz")
    rgb_dir = args.rgb_dir or (args.sample_dir / "per_frame_exports/rgb_pred")
    mask0 = args.mask0 or (args.work_dir / "sam2_ee_outputs/masks/mask_0000.png")
    rgb0 = rgb_dir / "frame_0000.png"

    ensure_mask(args, mask0, rgb_dir)

    z = np.load(pipeline_npz)
    if args.flow_key not in z.files:
        raise KeyError(f"flow_key={args.flow_key!r} not found in {pipeline_npz}; available keys: {z.files}")
    if args.point_key not in z.files:
        raise KeyError(f"point_key={args.point_key!r} not found in {pipeline_npz}; available keys: {z.files}")
    vae_flow = z[args.flow_key][0].astype(np.float64)
    vae_points0 = z[args.point_key][0, 0].astype(np.float64)
    sampled_indices = z["sampled_indices"].astype(np.int64)

    mask = np.asarray(Image.open(mask0).convert("L")) > 0
    valid = mask & np.isfinite(vae_points0).all(axis=-1) & np.isfinite(vae_flow).all(axis=(0, 3))
    if int(valid.sum()) <= 0:
        raise RuntimeError("No valid mask pixels after finite filtering.")

    flow_masked = vae_flow[:, valid, :]
    delta_pts_model_cv = flow_masked - flow_masked[0:1]
    ee_delta_model_cv = robust_aggregate(delta_pts_model_cv, args.agg, args.trim_frac)
    ee_delta_model_cv[0] = 0.0

    R_cam2zero, t_cam2zero, transform_raw = load_transform(args.cam2zero_json)
    ee_delta_zero = (R_cam2zero @ ee_delta_model_cv.T).T
    ee_delta_zero_smooth = moving_average(ee_delta_zero, args.smooth_window)
    ee_delta_zero_smooth[0] = 0.0

    with h5py.File(args.expert_hdf5, "r") as f:
        expert_pose = f["observations/ee_pose"][:].astype(np.float64)
        expert_attrs = {k: str(v) for k, v in f.attrs.items()}

    start_pose = expert_pose[0].copy()
    pred_pose49 = np.tile(start_pose[None, :], (len(sampled_indices), 1))
    pred_pose49[:, :3] = start_pose[:3][None, :] + float(args.scale) * ee_delta_zero_smooth

    export_frames = min(int(args.expert_cut), len(expert_pose))
    pred_pose130 = interpolate_pose(sampled_indices, pred_pose49, export_frames)

    compare_n = export_frames
    expert_cmp = expert_pose[:compare_n, :3]
    pred_cmp = pred_pose130[:, :3]
    expert_delta = expert_cmp - expert_cmp[0:1]
    pred_delta = pred_cmp - pred_cmp[0:1]
    err = np.linalg.norm(pred_cmp - expert_cmp, axis=1)

    attrs = {
        "generated_by": "run_scene_flow_to_arx_pipeline.py",
        "source_sample_dir": str(args.sample_dir),
        "source_pipeline_npz": str(pipeline_npz),
        "flow_key": args.flow_key,
        "point_key": args.point_key,
        "source_mask0": str(mask0),
        "source_cam2zero_json": str(args.cam2zero_json),
        "coordinate_convention": "ROS/ARX robot frame: +x forward, +y left, +z up; aligned to expert first EE pose",
        "model_scene_flow_convention": "metric scene flow aggregated from fixed mask pixels; applied directly through T_cam2end-zero",
        "transform_cam_convention": "T_cam2end-zero cam is used as the shared camera-to-zero rotation without an extra OpenCV/OpenGL flip",
        "scale": float(args.scale),
        "position_source": f"fixed-frame0-mask median scene flow from {args.flow_key}",
        "orientation_source": "constant expert first-frame roll,pitch,yaw placeholder",
        "non_ee_datasets": "qpos/qvel/effort/action/gripper are placeholders copied from the expert first row",
    }
    write_arx_like_hdf5(
        args.out_dir / "pred_ee_pose_arx_like_130.hdf5",
        args.expert_hdf5,
        pred_pose130,
        attrs | {"num_frames": int(len(pred_pose130)), "timeline": "interpolated to expert frames 0..129"},
    )

    save_overlay(rgb0, valid, args.out_dir / "frame0000_ee_mask_overlay.png")
    write_compare_csv(args.out_dir / "expert_vs_pred_scaled_compare_0_129.csv", expert_cmp, pred_cmp)

    np.savez_compressed(
        args.out_dir / "scene_flow_ee_pipeline_arrays.npz",
        sampled_indices=sampled_indices,
        valid_mask=valid.astype(np.uint8),
        ee_delta_model_cv=ee_delta_model_cv,
        ee_delta_zero=ee_delta_zero,
        ee_delta_zero_smooth=ee_delta_zero_smooth,
        pred_pose49=pred_pose49,
        pred_pose130=pred_pose130,
        expert_pose=expert_pose,
        scale=np.asarray(float(args.scale), dtype=np.float64),
        R_cam2zero=R_cam2zero,
        t_cam2zero=t_cam2zero,
    )

    summary = {
        "sample_dir": str(args.sample_dir),
        "pipeline_npz": str(pipeline_npz),
        "flow_key": args.flow_key,
        "point_key": args.point_key,
        "rgb_dir": str(rgb_dir),
        "mask0": str(mask0),
        "expert_hdf5": str(args.expert_hdf5),
        "cam2zero_json": str(args.cam2zero_json),
        "out_dir": str(args.out_dir),
        "scale": float(args.scale),
        "num_model_frames": int(len(sampled_indices)),
        "num_export_frames": int(len(pred_pose130)),
        "export_frame_range": [0, int(export_frames - 1)],
        "sampled_indices": sampled_indices.tolist(),
        "num_valid_mask_pixels": int(valid.sum()),
        "aggregation": args.agg,
        "smooth_window": int(args.smooth_window),
        "coordinate_chain": [
            f"{args.flow_key} fixed mask delta aggregated from the first-frame mask pixels",
            "delta vectors are rotated directly by T_cam2end-zero into robot zero/end/base frame",
            "scale delta by user scale",
            "align first frame to expert observations/ee_pose[0]",
        ],
        "compare_frames": [0, compare_n - 1],
        "metrics_pred_scaled_vs_expert_first_cut": metrics(pred_cmp, expert_cmp),
        "path_lengths_first_cut": {
            "expert": path_length(expert_delta),
            "pred_scaled": path_length(pred_delta),
        },
        "expert_attrs": expert_attrs,
        "transform_raw": transform_raw,
        "transform_rotation_matrix": R_cam2zero.tolist(),
        "transform_position": t_cam2zero.tolist(),
        "outputs": {
            "hdf5_130": str(args.out_dir / "pred_ee_pose_arx_like_130.hdf5"),
            "arrays_npz": str(args.out_dir / "scene_flow_ee_pipeline_arrays.npz"),
            "compare_csv": str(args.out_dir / "expert_vs_pred_scaled_compare_0_129.csv"),
            "mask_overlay": str(args.out_dir / "frame0000_ee_mask_overlay.png"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
