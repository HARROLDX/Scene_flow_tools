#!/usr/bin/env python3
"""Render dense RGB point-cloud trajectory visualizations for one sequence."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TRACEGEN_CV_TO_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

COLORS = {
    "expert": np.array([92, 184, 255], dtype=np.uint8),
    "ours": np.array([228, 38, 38], dtype=np.uint8),
    "tracegen": np.array([255, 118, 0], dtype=np.uint8),
}


def parse_args():
    base = Path("/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cube_10seq_50step")
    p = argparse.ArgumentParser("Pretty dense point-cloud trajectory renderer.")
    p.add_argument("--out_dir", type=Path, default=base / "ours_scene_flow/0_anchor_scene_flow")
    p.add_argument("--tracegen_npz", type=Path, default=Path("/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cube_10seq_dense/sequence_0/pipeline_outputs.npz"))
    p.add_argument("--tracegen_rgb0", type=Path, default=Path("/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cube_10seq_dense/sequence_0/per_frame_exports/rgb_gt/frame_0000.png"))
    p.add_argument("--tracegen_summary", type=Path, default=base / "tracegen_scene_flow/0_tracegen_scene_flow/summary.json")
    p.add_argument("--expert_hdf5", type=Path, default=Path("/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cube/gt_eepose/episode_0_ee_pose_arx.hdf5"))
    p.add_argument("--ours_hdf5", type=Path, default=base / "ours_scene_flow/0_anchor_scene_flow/pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5")
    p.add_argument("--tracegen_hdf5", type=Path, default=base / "tracegen_scene_flow/0_tracegen_scene_flow/tracegen_ee_pose_arx_like_130_inferred_optical_to_ros_gripper_closed_then_open.hdf5")
    p.add_argument("--max_points", type=int, default=500000)
    p.add_argument("--point_size_3d", type=float, default=1.18)
    p.add_argument("--point_size_2d", type=float, default=0.98)
    p.add_argument("--line_width", type=float, default=4.6)
    p.add_argument("--crop_margin_xy", type=float, default=0.24)
    p.add_argument("--crop_margin_z_below", type=float, default=0.13)
    p.add_argument("--crop_margin_z_above", type=float, default=0.20)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_pose(path):
    with h5py.File(path, "r") as f:
        return f["observations/ee_pose"][:130, :3].astype(np.float64)


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
    # Low-saturation bright pixels in clean table bands, away from the tray and gripper.
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    table_rgb = (hsv[..., 1] < 42) & (hsv[..., 2] > 118)
    right_table = (xx > int(0.68 * w)) & (yy > int(0.10 * h)) & (yy < int(0.82 * h))
    lower_table = (yy > int(0.72 * h)) & (xx > int(0.16 * w)) & (xx < int(0.82 * w))
    upper_table = (yy < int(0.18 * h)) & (xx > int(0.34 * w)) & (xx < int(0.90 * w))
    roi = (right_table | lower_table | upper_table) & table_rgb
    valid = roi & np.isfinite(pc_base_aligned).all(axis=-1)
    pts = pc_base_aligned[valid]
    if len(pts) < 500:
        valid = np.isfinite(pc_base_aligned).all(axis=-1)
        pts = pc_base_aligned[valid]
    if len(pts) > 15000:
        rng = np.random.default_rng(7)
        pts = pts[rng.choice(len(pts), 15000, replace=False)]
    center = np.median(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    R = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
    leveled = (R @ (pts - center).T).T
    return R, center, float(np.std(leveled[:, 2])), int(len(pts))


def set_clean_3d(ax, pts, trajs):
    all_pts = np.vstack([pts, *trajs])
    lo = np.nanpercentile(all_pts, 0.5, axis=0)
    hi = np.nanpercentile(all_pts, 99.5, axis=0)
    center = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.32)
    radius = span * 0.49
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.35, center[2] + radius * 0.65)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=30, azim=-58, roll=0)


def draw_traj_3d(ax, pts, color, label):
    rgb = np.asarray(color, dtype=np.float64) / 255.0
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, lw=4.8, alpha=0.98, label=label, solid_capstyle="round")
    ax.scatter(pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], color=[rgb], s=58, depthshade=False)
    ax.scatter(pts[-1:, 0], pts[-1:, 1], pts[-1:, 2], color=[rgb], s=78, depthshade=False, marker="X")


def render_3d(pc, colors, expert, ours, tracegen, out_path, args, highlight_pc=None, highlight_colors=None):
    fig = plt.figure(figsize=(10.5, 7.2), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=args.point_size_3d, alpha=0.62, marker="s", linewidths=0, depthshade=False)
    if highlight_pc is not None and len(highlight_pc):
        ax.scatter(
            highlight_pc[:, 0],
            highlight_pc[:, 1],
            highlight_pc[:, 2],
            c=highlight_colors,
            s=args.point_size_3d * 3.2,
            alpha=0.96,
            marker="s",
            linewidths=0,
            depthshade=False,
        )
    draw_traj_3d(ax, expert, COLORS["expert"], "expert")
    draw_traj_3d(ax, ours, COLORS["ours"], "ours")
    draw_traj_3d(ax, tracegen, COLORS["tracegen"], "TraceGen")
    set_clean_3d(ax, pc, [expert, ours, tracegen])
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6, borderpad=0.2, labelspacing=0.25)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def project_xy(points, axes, rotate_clockwise=False):
    a, b = axes
    xy = points[:, [a, b]].copy()
    if rotate_clockwise:
        xy = np.column_stack([xy[:, 1], -xy[:, 0]])
    return xy


def render_projection(
    pc,
    colors,
    expert,
    ours,
    tracegen,
    out_path,
    axes,
    title,
    args,
    rotate_clockwise=False,
    highlight_pc=None,
    highlight_colors=None,
):
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    pc_xy = project_xy(pc, axes, rotate_clockwise)
    ax.scatter(pc_xy[:, 0], pc_xy[:, 1], c=colors, s=args.point_size_2d, alpha=0.66, marker="s", linewidths=0)
    if highlight_pc is not None and len(highlight_pc):
        hi_xy = project_xy(highlight_pc, axes, rotate_clockwise)
        ax.scatter(hi_xy[:, 0], hi_xy[:, 1], c=highlight_colors, s=args.point_size_2d * 3.6, alpha=0.96, marker="s", linewidths=0)
    for pts, name, color in [(expert, "expert", COLORS["expert"]), (ours, "ours", COLORS["ours"]), (tracegen, "TraceGen", COLORS["tracegen"])]:
        rgb = color / 255.0
        pts_xy = project_xy(pts, axes, rotate_clockwise)
        ax.plot(pts_xy[:, 0], pts_xy[:, 1], color=rgb, lw=args.line_width, alpha=0.98, label=name, solid_capstyle="round")
        ax.scatter([pts_xy[0, 0]], [pts_xy[0, 1]], color=[rgb], s=58, zorder=5)
        ax.scatter([pts_xy[-1, 0]], [pts_xy[-1, 1]], color=[rgb], s=78, marker="X", zorder=5)
    all_pts = np.vstack([pc_xy, project_xy(expert, axes, rotate_clockwise), project_xy(ours, axes, rotate_clockwise), project_xy(tracegen, axes, rotate_clockwise)])
    lo = np.nanpercentile(all_pts, 1, axis=0)
    hi = np.nanpercentile(all_pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.35)
    r = span * 0.52
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12, pad=2)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6, borderpad=0.2, labelspacing=0.25)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def sample_polyline(points, spacing=0.004):
    chunks = []
    for a, b in zip(points[:-1], points[1:]):
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(dist / spacing)))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        chunks.append(a[None, :] * (1 - t) + b[None, :] * t)
    chunks.append(points[-1:])
    return np.vstack(chunks)


def write_ply(path, pc, colors, trajectories):
    traj_vertices = []
    traj_edges = []
    offset = len(pc)
    for name, pts, color in trajectories:
        dense = sample_polyline(pts)
        start = offset + len(traj_vertices)
        traj_vertices.append((dense, np.repeat(color[None, :], len(dense), axis=0)))
        traj_edges.extend((start + i, start + i + 1, color) for i in range(len(dense) - 1))
    if traj_vertices:
        tpts = np.vstack([x[0] for x in traj_vertices])
        tcols = np.vstack([x[1] for x in traj_vertices])
        verts = np.vstack([pc, tpts])
        cols = np.vstack([colors, tcols]).astype(np.uint8)
    else:
        verts = pc
        cols = colors.astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(traj_edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(verts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a, b, c in traj_edges:
            f.write(f"{a} {b} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def crop_points_around_trajectories(pc, colors, trajectories, args):
    traj = np.vstack(trajectories)
    lo = traj.min(axis=0) - np.array([args.crop_margin_xy, args.crop_margin_xy, args.crop_margin_z_below])
    hi = traj.max(axis=0) + np.array([args.crop_margin_xy, args.crop_margin_xy, args.crop_margin_z_above])
    keep = np.all((pc >= lo[None, :]) & (pc <= hi[None, :]), axis=1)
    if keep.sum() < 15000:
        return pc, colors
    return pc[keep], colors[keep]


def min_xy_distance_to_trajectories(pc, trajectories):
    traj_xy = np.vstack(trajectories)[:, :2]
    dist = np.empty(len(pc), dtype=np.float32)
    for start in range(0, len(pc), 20000):
        chunk = pc[start : start + 20000, :2]
        d2 = ((chunk[:, None, :] - traj_xy[None, :, :]) ** 2).sum(axis=2)
        dist[start : start + len(chunk)] = np.sqrt(d2.min(axis=1))
    return dist


def select_foreground_highlights(pc, colors, trajectories):
    if len(pc) == 0:
        return pc, colors
    rgb = colors.astype(np.float32)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    chroma = mx - mn
    red_cube = (r > 85) & (r > g * 1.25) & (r > b * 1.15)
    dark_robot = mx < 82
    bright_gripper = (mx > 135) & (chroma < 55)
    colored_foreground = chroma > 55
    near_traj = min_xy_distance_to_trajectories(pc, trajectories) < 0.18
    above_table = pc[:, 2] > 0.018
    keep = red_cube | (near_traj & above_table & (dark_robot | bright_gripper | colored_foreground))
    idx = np.flatnonzero(keep)
    if len(idx) > 90000:
        rng = np.random.default_rng(23)
        idx = rng.choice(idx, 90000, replace=False)
    return pc[idx], colors[idx].astype(np.float32) / 255.0


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.tracegen_npz)
    rgb = np.asarray(Image.open(args.tracegen_rgb0).convert("RGB"))
    pc_cv = z["rgbd_point_maps"][0].astype(np.float64)
    pc_base = np.einsum("ij,hwj->hwi", TRACEGEN_CV_TO_BASE, pc_cv)

    expert = load_pose(args.expert_hdf5)
    ours = load_pose(args.ours_hdf5)
    tracegen = load_pose(args.tracegen_hdf5)

    summary = json.loads(args.tracegen_summary.read_text())
    track_ids = np.asarray(summary["selected_track_ids"], dtype=np.int64)
    sparse0 = z["tracegen_sparse_points"][track_ids, 0, :].astype(np.float64)
    sparse0_base = (TRACEGEN_CV_TO_BASE @ sparse0.T).T
    align = expert[0] - np.nanmedian(sparse0_base, axis=0)
    pc_base_aligned = pc_base + align[None, None, :]

    finite = np.isfinite(pc_base_aligned).all(axis=-1)
    flat_pts = pc_base_aligned[finite]
    flat_rgb = rgb[finite].astype(np.float32) / 255.0

    R_level, table_center, table_std, table_n = fit_table_level_transform(pc_base_aligned, rgb)

    def level(x):
        return (R_level @ (x - table_center).T).T

    pc_level = level(flat_pts)
    expert_level = level(expert)
    ours_level = level(ours)
    tracegen_level = level(tracegen)
    rgb_u8 = np.clip(flat_rgb * 255, 0, 255).astype(np.uint8)

    pc_level, rgb_u8 = crop_points_around_trajectories(
        pc_level,
        rgb_u8,
        [expert_level, ours_level, tracegen_level],
        args,
    )
    if len(pc_level) > args.max_points:
        rng = np.random.default_rng(11)
        pick = rng.choice(len(pc_level), args.max_points, replace=False)
        pc_level = pc_level[pick]
        rgb_u8 = rgb_u8[pick]
    flat_rgb = rgb_u8.astype(np.float32) / 255.0
    highlight_pc, highlight_rgb = select_foreground_highlights(
        pc_level,
        rgb_u8,
        [expert_level, ours_level, tracegen_level],
    )

    render_3d(pc_level, flat_rgb, expert_level, ours_level, tracegen_level, args.out_dir / "pretty_pointcloud_trajectory_3d.png", args, highlight_pc, highlight_rgb)
    top_path = args.out_dir / "pretty_pointcloud_trajectory_top.png"
    render_projection(pc_level, flat_rgb, expert_level, ours_level, tracegen_level, top_path, (0, 1), "", args, True, highlight_pc, highlight_rgb)
    render_projection(pc_level, flat_rgb, expert_level, ours_level, tracegen_level, args.out_dir / "pretty_pointcloud_trajectory_side.png", (0, 2), "", args, False, highlight_pc, highlight_rgb)
    write_ply(
        args.out_dir / "pretty_pointcloud_trajectory.ply",
        pc_level,
        rgb_u8,
        [
            ("expert", expert_level, COLORS["expert"]),
            ("ours", ours_level, COLORS["ours"]),
            ("TraceGen", tracegen_level, COLORS["tracegen"]),
        ],
    )

    meta = {
        "pointcloud_source": str(args.tracegen_npz),
        "num_display_points": int(len(pc_level)),
        "tracegen_cv_to_base": TRACEGEN_CV_TO_BASE.tolist(),
        "alignment": "selected TraceGen track median at frame0 aligned to expert first EE position",
        "table_leveling_std": table_std,
        "num_table_points": table_n,
        "outputs": {
            "pretty_3d": str(args.out_dir / "pretty_pointcloud_trajectory_3d.png"),
            "pretty_top": str(args.out_dir / "pretty_pointcloud_trajectory_top.png"),
            "pretty_side": str(args.out_dir / "pretty_pointcloud_trajectory_side.png"),
            "pretty_ply": str(args.out_dir / "pretty_pointcloud_trajectory.ply"),
        },
    }
    (args.out_dir / "pretty_pointcloud_trajectory_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
