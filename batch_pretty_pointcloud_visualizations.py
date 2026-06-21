#!/usr/bin/env python3
"""Render pretty RGB point-cloud trajectory figures for all ARX scene-flow runs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


TASKS = {
    "grasp_cube": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cube_10seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cube_10seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cube/gt_eepose",
        "num": 10,
    },
    "green_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_green_tray_5seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/green_tray_5seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/green_tray/gt_eepose",
        "num": 5,
    },
    "yellow_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_yellow_tray_5seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/yellow_tray_5seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/yellow_tray/gt_eepose",
        "num": 5,
    },
    "grasp_cup": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cup_10seq_50step",
        "tracegen": "/mnt/workspace/wudong/TraceGen/arx_tracegen_rrd_output/grasp_cup_10seq_dense",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cup/gt_eepose",
        "num": 10,
    },
}


def parse_args():
    p = argparse.ArgumentParser("Batch-render pretty point-cloud trajectory visualizations.")
    p.add_argument("--work_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--output_subdir", default=None, help="Write pretty outputs under each sequence directory subfolder.")
    return p.parse_args()


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def process_one(args, task_name, cfg, idx):
    base = Path(cfg["base"])
    tracegen_dir = Path(cfg["tracegen"]) / f"sequence_{idx}"
    seq_dir = base / cfg.get("ours_root", "ours_scene_flow") / f"{idx}_anchor_scene_flow"
    out_dir = seq_dir / args.output_subdir if args.output_subdir else seq_dir
    tracegen_out = base / "tracegen_scene_flow" / f"{idx}_tracegen_scene_flow"
    expert_hdf5 = Path(cfg["gt"]) / f"episode_{idx}_ee_pose_arx.hdf5"

    paths = {
        "out_dir": out_dir,
        "tracegen_npz": tracegen_dir / "pipeline_outputs.npz",
        "tracegen_rgb0": tracegen_dir / "per_frame_exports/rgb_gt/frame_0000.png",
        "tracegen_summary": tracegen_out / "summary.json",
        "expert_hdf5": expert_hdf5,
        "ours_hdf5": seq_dir / "pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5",
        "tracegen_hdf5": tracegen_out / "tracegen_ee_pose_arx_like_130_inferred_optical_to_ros_gripper_closed_then_open.hdf5",
    }
    for name, path in paths.items():
        if name != "out_dir" and not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    run(
        [
            args.python,
            args.work_dir / "make_pretty_pointcloud_trajectory.py",
            "--out_dir",
            paths["out_dir"],
            "--tracegen_npz",
            paths["tracegen_npz"],
            "--tracegen_rgb0",
            paths["tracegen_rgb0"],
            "--tracegen_summary",
            paths["tracegen_summary"],
            "--expert_hdf5",
            paths["expert_hdf5"],
            "--ours_hdf5",
            paths["ours_hdf5"],
            "--tracegen_hdf5",
            paths["tracegen_hdf5"],
        ]
    )
    return {k: str(v) for k, v in paths.items()}


def main():
    args = parse_args()
    all_results = {}
    for task_name in args.tasks:
        cfg = TASKS[task_name]
        start = 0 if args.start is None else max(0, args.start)
        end = int(cfg["num"]) - 1 if args.end is None else min(int(cfg["num"]) - 1, args.end)
        results = []
        errors = []
        for idx in range(start, end + 1):
            print(f"\n=== Pretty visualization {task_name} sequence {idx} ===", flush=True)
            try:
                results.append(process_one(args, task_name, cfg, idx))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                errors.append({"idx": idx, "error": repr(exc)})
                print(f"ERROR: {task_name} {idx}: {exc!r}", flush=True)
        summary = {
            "task": task_name,
            "base": cfg["base"],
            "tracegen_root": cfg["tracegen"],
            "gt": cfg["gt"],
            "results": results,
            "errors": errors,
        }
        out_root = Path(cfg["base"]) / cfg.get("ours_root", "ours_scene_flow")
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "pretty_pointcloud_batch_summary.json").write_text(json.dumps(summary, indent=2))
        all_results[task_name] = summary
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
