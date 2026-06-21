#!/usr/bin/env python3
"""Batch build TraceGen ARX-like trajectories for grasp-cup sequences."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Kept as a lightweight helper module for TraceGen conversion.
PROMPTS = {}


TRACEGEN_CV_TO_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args():
    base = Path("/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cup_10seq_50step")
    tracegen = Path("/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cup_10seq_dense")
    gt = Path("/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cup/gt_eepose")
    p = argparse.ArgumentParser("Batch process TraceGen grasp-cup trajectories.")
    p.add_argument("--base_dir", type=Path, default=base)
    p.add_argument("--tracegen_root", type=Path, default=tracegen)
    p.add_argument("--work_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--gt_dir", type=Path, default=gt)
    p.add_argument("--ours_root", type=Path, default=base / "ours_scene_flow")
    p.add_argument("--out_root", type=Path, default=base / "tracegen_scene_flow")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--expert_cut", type=int, default=130)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--min_track_pixels", type=int, default=20)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=9)
    p.add_argument("--closed_value", type=float, default=0.0)
    p.add_argument("--open_value", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no_overwrite", action="store_false", dest="overwrite")
    p.add_argument("--skip_sam", action="store_true", help="Reuse existing per-output SAM2 mask if present.")
    return p.parse_args()


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def quat_xyzw_to_rot(q):
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_transform(path):
    data = json.loads(path.read_text())
    return quat_xyzw_to_rot(data["orientation"]), np.asarray(data["position"], dtype=np.float64), data


def moving_average(x, window):
    window = int(window)
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([xp[i : i + window].mean(axis=0) for i in range(len(x))], axis=0)


def interp_by_frame_indices(frame_indices, values, target_n):
    target = np.arange(int(target_n), dtype=np.float64)
    out = np.zeros((target_n, values.shape[1]), dtype=np.float64)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(target, frame_indices.astype(np.float64), values[:, axis])
    return out


def scale_prompt(prompt, src_wh=(560.0, 448.0), dst_wh=(640.0, 480.0)):
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]

    def sp(p):
        return [float(p[0]) * sx, float(p[1]) * sy]

    out = dict(prompt)
    out["points"] = [sp(p) for p in prompt["points"]]
    out["box"] = [prompt["box"][0] * sx, prompt["box"][1] * sy, prompt["box"][2] * sx, prompt["box"][3] * sy]
    out["hard_roi"] = [
        prompt["hard_roi"][0] * sx,
        prompt["hard_roi"][1] * sy,
        prompt["hard_roi"][2] * sx,
        prompt["hard_roi"][3] * sy,
    ]
    out["frame_idx"] = 0
    out["exclude_saturated"] = False
    return out


def select_tracks_from_mask(z, mask, min_track_pixels):
    dense_idx0 = z["tracegen_dense_track_index"][0]
    dense_valid0 = z["tracegen_dense_valid_mask"][0]
    ids = dense_idx0[mask & dense_valid0]
    ids = ids[ids >= 0]
    uniq, counts = np.unique(ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq, counts = uniq[order], counts[order]
    keep = counts >= int(min_track_pixels)
    selected = uniq[keep]
    selected_counts = counts[keep]
    if selected.size == 0:
        selected = uniq[: min(6, uniq.size)]
        selected_counts = counts[: min(6, counts.size)]
    return selected.astype(np.int64), selected_counts.astype(np.int64), uniq.astype(np.int64), counts.astype(np.int64)


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


def write_arx_like_hdf5(path, template_hdf5, ee_pose, attrs, closed_value=0.0, open_value=5.0):
    n = int(len(ee_pose))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(template_hdf5, "r") as src, h5py.File(path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for k, v in attrs.items():
            dst.attrs[k] = json.dumps(v) if isinstance(v, (dict, list, tuple)) else v

        obs = dst.create_group("observations")
        ee_pose32 = ee_pose.astype(np.float32)
        obs.create_dataset("ee_pose", data=ee_pose32)
        dst.create_dataset("action_ee_pose", data=ee_pose32)

        gripper = np.full((n,), np.float32(closed_value), dtype=np.float32)
        gripper[-1] = np.float32(open_value)
        obs.create_dataset("gripper", data=gripper)
        dst.create_dataset("action_gripper", data=gripper)

        qpos0 = src["observations/qpos"][0].astype(np.float32)
        qvel0 = src["observations/qvel"][0].astype(np.float32)
        effort0 = src["observations/effort"][0].astype(np.float32)
        action0 = src["action"][0].astype(np.float32)
        qpos = np.tile(qpos0[None, :], (n, 1))
        action = np.tile(action0[None, :], (n, 1))
        qpos[:, -1] = gripper
        action[:, -1] = gripper
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=np.tile(qvel0[None, :], (n, 1)))
        obs.create_dataset("effort", data=np.tile(effort0[None, :], (n, 1)))
        dst.create_dataset("action", data=action)

        dpos = np.diff(ee_pose32[:, :3], axis=0, prepend=ee_pose32[0:1, :3])
        action_delta = np.zeros((n, 5), dtype=np.float32)
        action_delta[:, :3] = dpos
        action_delta[:, 4] = np.diff(gripper, prepend=gripper[0])
        dst.create_dataset("action_delta_ee", data=action_delta)

        dst.attrs["gripper_policy"] = f"closed for frames 0..{n - 2}, open on final frame {n - 1}"
        dst.attrs["gripper_closed_value"] = float(closed_value)
        dst.attrs["gripper_open_value"] = float(open_value)


def save_mask_overlay(rgb_path, mask, track_uv0, selected_ids, out_path):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([0.0, 255.0, 80.0])
    im = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for tid in selected_ids:
        x, y = track_uv0[int(tid)]
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 0), outline=(0, 0, 0))
        d.text((x + 6, y - 6), str(int(tid)), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(out_path)


def set_equal_3d(ax, points, min_radius=0.04):
    lo, hi = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float((hi - lo).max()) * 0.55, min_radius)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_3d(expert, tracegen, ours, out_path, title):
    fig = plt.figure(figsize=(8, 7), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(expert[:, 0], expert[:, 1], expert[:, 2], color="black", lw=2.5, label="expert")
    ax.plot(ours[:, 0], ours[:, 1], ours[:, 2], color="tab:blue", lw=2.0, label="ours")
    ax.plot(tracegen[:, 0], tracegen[:, 1], tracegen[:, 2], color="tab:orange", lw=2.0, label="TraceGen")
    ax.scatter(*expert[0], c="limegreen", s=48, label="start")
    ax.scatter(*expert[-1], c="crimson", s=48, label="expert end")
    set_equal_3d(ax, np.vstack([expert, tracegen, ours]))
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_zlabel("z up (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_projections(expert_delta, tracegen_delta, ours_delta, out_path, title):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=180)
    specs = [
        (0, 1, "top view x-y", "x forward delta (m)", "y left delta (m)"),
        (0, 2, "side view x-z", "x forward delta (m)", "z up delta (m)"),
        (1, 2, "side view y-z", "y left delta (m)", "z up delta (m)"),
    ]
    for ax, (a, b, title_i, xlabel, ylabel) in zip(axs, specs):
        ax.plot(expert_delta[:, a], expert_delta[:, b], color="black", lw=2.5, label="expert")
        ax.plot(ours_delta[:, a], ours_delta[:, b], color="tab:blue", lw=2.0, label="ours")
        ax.plot(tracegen_delta[:, a], tracegen_delta[:, b], color="tab:orange", lw=2.0, label="TraceGen")
        ax.scatter([0.0], [0.0], c="limegreen", s=34, zorder=4)
        ax.axhline(0.0, color="0.6", lw=0.8, alpha=0.45)
        ax.axvline(0.0, color="0.6", lw=0.8, alpha=0.45)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_title(title_i)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axs[0].legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rotation_from_vectors(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-10:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def fit_table_level_transform(pc_base_aligned, rgb):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    table_rgb = (hsv[..., 1] < 38) & (hsv[..., 2] > 125)
    # Prefer clean table bands away from the cup, tray, robot, and cable.
    right_table = (xx > int(0.70 * w)) & (yy > int(0.12 * h)) & (yy < int(0.82 * h))
    lower_table = (yy > int(0.72 * h)) & (xx > int(0.18 * w)) & (xx < int(0.82 * w))
    upper_table = (yy < int(0.18 * h)) & (xx > int(0.35 * w)) & (xx < int(0.90 * w))
    roi = (right_table | lower_table | upper_table) & table_rgb
    valid = roi & np.isfinite(pc_base_aligned).all(axis=-1)
    pts = pc_base_aligned[valid]
    if len(pts) < 200:
        valid = np.isfinite(pc_base_aligned).all(axis=-1)
        pts = pc_base_aligned[valid]
    if len(pts) > 12000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 12000, replace=False)]
    center = np.median(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    R = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
    leveled = (R @ (pts - center).T).T
    return R, center, float(np.std(leveled[:, 2])), int(len(pts))


def plot_pointcloud_3d(pc_xyz, colors, expert, tracegen, ours, out_path, title):
    fig = plt.figure(figsize=(9, 8), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c=colors, s=0.35, alpha=0.35, linewidths=0)
    ax.plot(expert[:, 0], expert[:, 1], expert[:, 2], color="black", lw=2.6, label="expert")
    ax.plot(ours[:, 0], ours[:, 1], ours[:, 2], color="tab:blue", lw=2.1, label="ours")
    ax.plot(tracegen[:, 0], tracegen[:, 1], tracegen[:, 2], color="tab:orange", lw=2.1, label="TraceGen")
    ax.scatter(*expert[0], c="limegreen", s=50, label="start")
    set_equal_3d(ax, np.vstack([pc_xyz, expert, tracegen, ours]), min_radius=0.12)
    ax.set_xlabel("x display (m)")
    ax.set_ylabel("y display (m)")
    ax.set_zlabel("height above fitted table (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pointcloud_projections(pc_xyz, colors, expert, tracegen, ours, out_path, title):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180)
    specs = [(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]
    for ax, (a, b, name) in zip(axs, specs):
        ax.scatter(pc_xyz[:, a], pc_xyz[:, b], c=colors, s=0.35, alpha=0.25, linewidths=0)
        ax.plot(expert[:, a], expert[:, b], color="black", lw=2.3, label="expert")
        ax.plot(ours[:, a], ours[:, b], color="tab:blue", lw=2.0, label="ours")
        ax.plot(tracegen[:, a], tracegen[:, b], color="tab:orange", lw=2.0, label="TraceGen")
        ax.scatter([expert[0, a]], [expert[0, b]], c="limegreen", s=34)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        ax.set_title(name)
    axs[0].legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_pointcloud_visuals(z, rgb0_path, track_ids, expert, tracegen, ours, out_dir):
    rgb = np.asarray(Image.open(rgb0_path).convert("RGB"))
    pc_cv = z["rgbd_point_maps"][0].astype(np.float64)
    pc_base = np.einsum("ij,hwj->hwi", TRACEGEN_CV_TO_BASE, pc_cv)
    sparse0 = z["tracegen_sparse_points"][track_ids, 0, :].astype(np.float64)
    sparse0_base = (TRACEGEN_CV_TO_BASE @ sparse0.T).T
    align = expert[0, :3] - np.nanmedian(sparse0_base, axis=0)
    pc_base_aligned = pc_base + align[None, None, :]

    finite = np.isfinite(pc_base_aligned).all(axis=-1)
    flat_pts = pc_base_aligned[finite]
    flat_rgb = rgb[finite].astype(np.float32) / 255.0
    if len(flat_pts) > 45000:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(flat_pts), 45000, replace=False)
        flat_pts = flat_pts[pick]
        flat_rgb = flat_rgb[pick]

    R_level, table_center, table_std, table_n = fit_table_level_transform(pc_base_aligned, rgb)

    def level(x):
        return (R_level @ (x - table_center).T).T

    pc_level = level(flat_pts)
    expert_level = level(expert)
    trace_level = level(tracegen)
    ours_level = level(ours)

    plot_pointcloud_3d(
        pc_level,
        flat_rgb,
        expert_level,
        trace_level,
        ours_level,
        out_dir / "expert_vs_ours_vs_tracegen_pointcloud_aligned_3d.png",
        "Table-leveled point cloud with trajectories",
    )
    plot_pointcloud_projections(
        pc_level,
        flat_rgb,
        expert_level,
        trace_level,
        ours_level,
        out_dir / "expert_vs_ours_vs_tracegen_pointcloud_aligned_projections.png",
        "Table-leveled point cloud projections",
    )
    summary = {
        "visualization_only": True,
        "tracegen_cv_to_base": TRACEGEN_CV_TO_BASE.tolist(),
        "pointcloud_alignment": "selected TraceGen gripper track center at frame0 aligned to expert first EE position",
        "table_leveling": "fit plane on low-saturation bright table ROI, rotate all displayed data so table normal is +z",
        "table_z_after_leveling_std": table_std,
        "num_table_points": table_n,
        "num_display_points": int(len(flat_pts)),
    }
    (out_dir / "table_level_visualization_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def process_one(args, idx):
    tracegen_dir = args.tracegen_root / f"sequence_{idx}"
    pipeline_npz = tracegen_dir / "pipeline_outputs.npz"
    rgb_dir = tracegen_dir / "per_frame_exports/rgb_gt"
    rgb0 = rgb_dir / "frame_0000.png"
    expert_hdf5 = args.gt_dir / f"episode_{idx}_ee_pose_arx.hdf5"
    ours_hdf5 = args.ours_root / f"{idx}_anchor_scene_flow/pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5"
    out_dir = args.out_root / f"{idx}_tracegen_scene_flow"
    mask0 = out_dir / "sam2_ee_outputs/masks/mask_0000.png"

    for p in [tracegen_dir, pipeline_npz, rgb0, expert_hdf5, ours_hdf5, args.work_dir / "T_cam2end-zero.json"]:
        if not p.exists():
            raise FileNotFoundError(p)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_dir = args.out_root / "sam2_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt = scale_prompt(PROMPTS[idx])
    prompt_path = prompt_dir / f"prompt_seq_{idx}_tracegen.json"
    prompt_path.write_text(json.dumps(prompt, indent=2))

    if not args.skip_sam or not mask0.exists():
        run(
            [
                "python3",
                args.work_dir / "segment_ee_with_sam2.py",
                "--rgb_dir",
                rgb_dir,
                "--out_dir",
                out_dir / "sam2_ee_outputs",
                "--prompt_json",
                prompt_path,
                "--max_area",
                "14000",
            ]
        )

    z = np.load(pipeline_npz)
    mask = np.asarray(Image.open(mask0).convert("L")) > 0
    track_ids, track_counts, all_ids, all_counts = select_tracks_from_mask(z, mask, args.min_track_pixels)
    if track_ids.size == 0:
        raise RuntimeError(f"No TraceGen tracks selected for sequence {idx}")

    sparse_points = z["tracegen_sparse_points"].astype(np.float64)
    frame_indices = z["frame_indices"].astype(np.int64)
    track_points = sparse_points[track_ids]
    delta_cv = track_points - track_points[:, 0:1, :]
    ee_delta_cv = np.nanmedian(delta_cv, axis=0)
    ee_delta_cv[0] = 0.0

    R_cam2zero, t_cam2zero, transform_raw = load_transform(args.work_dir / "T_cam2end-zero.json")
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0])
    old_delta = (R_cam2zero @ (opencv_to_opengl @ ee_delta_cv.T)).T
    old_delta = moving_average(old_delta, args.smooth_window)
    old_delta[0] = 0.0

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

    old_pose = pose_from_delta(old_delta)
    inferred_pose = pose_from_delta(inferred_delta)
    with h5py.File(ours_hdf5, "r") as f:
        ours_pose = f["observations/ee_pose"][:export_n].astype(np.float64)

    old_hdf5 = out_dir / "tracegen_ee_pose_arx_like_130_gripper_closed_then_open.hdf5"
    inferred_hdf5 = out_dir / "tracegen_ee_pose_arx_like_130_inferred_optical_to_ros_gripper_closed_then_open.hdf5"
    common_attrs = {
        "generated_by": "batch_cup_tracegen_scene_flow.py",
        "source_tracegen_dir": str(tracegen_dir),
        "source_pipeline_npz": str(pipeline_npz),
        "source_mask0": str(mask0),
        "selected_track_ids": track_ids.tolist(),
        "selected_track_pixel_counts": track_counts.tolist(),
        "scale": float(args.scale),
        "orientation_source": "constant expert first-frame roll,pitch,yaw placeholder",
    }
    write_arx_like_hdf5(
        old_hdf5,
        expert_hdf5,
        old_pose,
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
    old_cmp = old_pose[:, :3]
    plot_3d(expert_cmp, inferred_cmp, ours_cmp, out_dir / "expert_vs_ours_vs_tracegen_3d.png", f"Cup {idx}: expert vs ours vs TraceGen inferred")
    plot_delta_projections(
        expert_cmp - expert_cmp[0:1],
        inferred_cmp - inferred_cmp[0:1],
        ours_cmp - ours_cmp[0:1],
        out_dir / "expert_vs_ours_vs_tracegen_ros_2d.png",
        f"Cup {idx}: ROS delta projections",
    )
    table_summary = make_pointcloud_visuals(z, rgb0, track_ids, expert_cmp, inferred_cmp, ours_cmp, out_dir)

    np.savez_compressed(
        out_dir / "tracegen_ee_pipeline_arrays.npz",
        selected_track_ids=track_ids,
        selected_track_pixel_counts=track_counts,
        all_mask_track_ids=all_ids,
        all_mask_track_pixel_counts=all_counts,
        frame_indices=frame_indices,
        mask0=mask.astype(np.uint8),
        ee_delta_cv=ee_delta_cv,
        old_delta=old_delta,
        inferred_delta=inferred_delta,
        old_pose130=old_pose,
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
        "idx": idx,
        "tracegen_dir": str(tracegen_dir),
        "pipeline_npz": str(pipeline_npz),
        "mask0": str(mask0),
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
            "tracegen_old_vs_expert": metrics(old_cmp, expert_cmp),
            "tracegen_inferred_vs_expert": metrics(inferred_cmp, expert_cmp),
        },
        "path_lengths_first_cut": {
            "expert": path_length(expert_cmp - expert_cmp[0:1]),
            "ours": path_length(ours_cmp - ours_cmp[0:1]),
            "tracegen_old": path_length(old_cmp - old_cmp[0:1]),
            "tracegen_inferred": path_length(inferred_cmp - inferred_cmp[0:1]),
        },
        "transform_raw": transform_raw,
        "tracegen_inferred_transform": {
            "mapping": "base_x=cv_z, base_y=-cv_x, base_z=-cv_y",
            "matrix": TRACEGEN_CV_TO_BASE.tolist(),
        },
        "table_level_visualization": table_summary,
        "outputs": {
            "legacy_hdf5": str(old_hdf5),
            "inferred_hdf5": str(inferred_hdf5),
            "arrays_npz": str(out_dir / "tracegen_ee_pipeline_arrays.npz"),
            "mask_track_overlay": str(out_dir / "frame0000_ee_mask_tracks_overlay.png"),
            "plot_3d": str(out_dir / "expert_vs_ours_vs_tracegen_3d.png"),
            "plot_2d": str(out_dir / "expert_vs_ours_vs_tracegen_ros_2d.png"),
            "pointcloud_3d": str(out_dir / "expert_vs_ours_vs_tracegen_pointcloud_aligned_3d.png"),
            "pointcloud_projections": str(out_dir / "expert_vs_ours_vs_tracegen_pointcloud_aligned_projections.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "tracegen_inferred_transform_summary.json").write_text(
        json.dumps(summary["tracegen_inferred_transform"] | {"metrics": summary["metrics"]["tracegen_inferred_vs_expert"]}, indent=2)
    )
    (out_dir / "README.md").write_text(
        f"# {idx}_tracegen_scene_flow\n\n"
        "TraceGen converted to 130-frame ARX-like EE trajectories for grasp_cup. "
        "The inferred HDF5 uses fixed sparse tracks selected by the first-frame EE mask and "
        "`base_x=cv_z, base_y=-cv_x, base_z=-cv_y`, then aligns frame 0 to the expert trajectory.\n"
    )
    return summary


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in range(int(args.start), int(args.end) + 1):
        print(f"\n=== Processing TraceGen cup sequence {idx} ===", flush=True)
        if idx not in PROMPTS:
            raise KeyError(f"No prompt configured for sequence {idx}")
        results.append(process_one(args, idx))

    batch_summary = {
        "task": "grasp_cup_tracegen",
        "tracegen_root": str(args.tracegen_root),
        "out_root": str(args.out_root),
        "expert_dir": str(args.gt_dir),
        "ours_root": str(args.ours_root),
        "scale": float(args.scale),
        "expert_cut": int(args.expert_cut),
        "results": results,
    }
    (args.out_root / "tracegen_cup_0_6_batch_summary.json").write_text(json.dumps(batch_summary, indent=2))
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
