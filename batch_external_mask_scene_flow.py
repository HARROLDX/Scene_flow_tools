#!/usr/bin/env python3
"""Batch scene-flow trajectories using externally supplied first-frame masks."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


TASKS = {
    "grasp_cube": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cube_10seq_50step",
        "sample_name": "grasp_cube",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cube/gt_eepose",
        "num": 10,
        "skip": [],
    },
    "grasp_cup": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_grasp_cup_10seq_50step",
        "sample_name": "grasp_cup",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/grasp_cup/gt_eepose",
        "num": 10,
        "skip": [],
    },
    "green_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_green_tray_5seq_50step",
        "sample_name": "green_tray",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/green_tray/gt_eepose",
        "num": 5,
        "skip": [],
    },
    "yellow_tray": {
        "base": "/mnt/workspace/wudong/pipeline_inference_package/arx_data_test1_yellow_tray_5seq_50step",
        "sample_name": "yellow_tray",
        "gt": "/mnt/workspace/wudong/Gen3R/data/ARX_data_test1/yellow_tray/gt_eepose",
        "num": 5,
        "skip": [],
    },
}


def parse_args():
    p = argparse.ArgumentParser("Run ours scene-flow pipeline with external first-frame masks.")
    p.add_argument("--work_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--mask_root", type=Path, default=Path("/mnt/workspace/wudong/Gen3R/data/ARX_masks"))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--expert_cut", type=int, default=130)
    p.add_argument("--closed_value", type=float, default=0.0)
    p.add_argument("--open_value", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no_overwrite", action="store_false", dest="overwrite")
    return p.parse_args()


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resize_mask(mask_path, rgb0_path, dst_path):
    rgb = Image.open(rgb0_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst_path)
    return int(np.count_nonzero(np.asarray(mask) > 0)), list(mask.size)


def finalize_gripper(src_hdf5, dst_hdf5, closed_value=0.0, open_value=5.0):
    src_hdf5 = Path(src_hdf5)
    dst_hdf5 = Path(dst_hdf5)
    if dst_hdf5.exists():
        dst_hdf5.unlink()
    with h5py.File(src_hdf5, "r") as fin, h5py.File(dst_hdf5, "w") as fout:
        def copy_item(name, obj):
            if isinstance(obj, h5py.Group):
                fout.require_group(name)
            elif isinstance(obj, h5py.Dataset):
                fout.create_dataset(name, data=obj[...])

        fin.visititems(copy_item)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        n = fout["observations/ee_pose"].shape[0]
        gripper = np.full((n,), np.float32(closed_value), dtype=np.float32)
        gripper[-1] = np.float32(open_value)
        fout["observations/gripper"][...] = gripper
        fout["action_gripper"][...] = gripper

        qpos = fout["observations/qpos"][...]
        qpos[:, -1] = gripper
        fout["observations/qpos"][...] = qpos

        action = fout["action"][...]
        action[:, -1] = gripper
        fout["action"][...] = action

        delta = fout["action_delta_ee"][...]
        delta[:, 4] = np.diff(gripper, prepend=gripper[0])
        fout["action_delta_ee"][...] = delta

        fout.attrs["gripper_policy"] = f"closed for frames 0..{n - 2}, open on final frame {n - 1}"
        fout.attrs["gripper_closed_value"] = float(closed_value)
        fout.attrs["gripper_open_value"] = float(open_value)
        fout.attrs["gripper_value_source"] = "external-mask batch: closed=0.0, final open=5.0"


def process_one(args, task_name, cfg, idx):
    base = Path(cfg["base"])
    sample_name = cfg["sample_name"]
    sample_dir = base / f"sample_{idx:02d}_{sample_name}_video_{idx}"
    gt_dir = Path(cfg["gt"])
    expert_hdf5 = gt_dir / f"episode_{idx}_ee_pose_arx.hdf5"
    out_dir = base / "ours_scene_flow" / f"{idx}_anchor_scene_flow"
    pipeline_npz = sample_dir / "pipeline_outputs.npz"
    rgb_dir = sample_dir / "per_frame_exports/rgb_pred"
    rgb0 = rgb_dir / "frame_0000.png"
    external_mask = args.mask_root / task_name / f"{idx}_first_frame_mask.png"

    for path in [sample_dir, pipeline_npz, rgb0, expert_hdf5, external_mask, args.work_dir / "T_cam2end-zero.json"]:
        if not path.exists():
            raise FileNotFoundError(path)

    old_summary = load_json(out_dir / "summary.json")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask0 = out_dir / "external_mask_0000_resized.png"
    mask_pixels, mask_size = resize_mask(external_mask, rgb0, mask0)

    run(
        [
            "python3",
            args.work_dir / "run_scene_flow_to_arx_pipeline.py",
            "--sample_dir",
            sample_dir,
            "--pipeline_npz",
            pipeline_npz,
            "--rgb_dir",
            rgb_dir,
            "--mask0",
            mask0,
            "--expert_hdf5",
            expert_hdf5,
            "--cam2zero_json",
            args.work_dir / "T_cam2end-zero.json",
            "--flow_key",
            "vae_metric_scene_flow",
            "--point_key",
            "vae_metric_point_maps",
            "--scale",
            str(args.scale),
            "--expert_cut",
            str(args.expert_cut),
            "--out_dir",
            out_dir,
            "--skip_sam_if_missing",
        ]
    )

    final_hdf5 = out_dir / "pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5"
    finalize_gripper(
        out_dir / "pred_ee_pose_arx_like_130.hdf5",
        final_hdf5,
        args.closed_value,
        args.open_value,
    )

    summary = load_json(out_dir / "summary.json")
    summary["external_mask"] = {
        "source": str(external_mask),
        "resized": str(mask0),
        "resized_size_wh": mask_size,
        "resized_pixels": mask_pixels,
    }
    summary["previous_summary_before_external_mask_overwrite"] = old_summary
    summary["outputs"]["hdf5_130_gripper_closed_then_open"] = str(final_hdf5)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    all_results = {}
    for task_name in args.tasks:
        cfg = TASKS[task_name]
        results = []
        for idx in range(int(cfg["num"])):
            if idx in set(cfg.get("skip", [])):
                print(f"\n=== Skipping {task_name} sequence {idx} by config ===", flush=True)
                continue
            print(f"\n=== Processing {task_name} sequence {idx} with external mask ===", flush=True)
            results.append(process_one(args, task_name, cfg, idx))
        out_root = Path(cfg["base"]) / "ours_scene_flow"
        batch = {
            "task": task_name,
            "base": cfg["base"],
            "mask_root": str(args.mask_root / task_name),
            "scale": float(args.scale),
            "expert_cut": int(args.expert_cut),
            "skipped_indices": cfg.get("skip", []),
            "results": results,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "external_mask_batch_summary.json").write_text(json.dumps(batch, indent=2))
        all_results[task_name] = batch
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
