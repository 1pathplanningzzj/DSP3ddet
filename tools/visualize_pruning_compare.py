import argparse
import contextlib
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dspdet3d  # noqa: F401


DEFAULT_BASELINE_CONFIG = REPO_ROOT / 'work_dirs/scannet_baseline_1gpu_bs4_gpu2/dspdet3d_scannet-3d-22class.py'
DEFAULT_BASELINE_CKPT = REPO_ROOT / 'work_dirs/scannet_baseline_1gpu_bs4_gpu2/latest.pth'
DEFAULT_GAUSSIAN_CONFIG = REPO_ROOT / 'work_dirs/scannet_gaussian_final_1gpu_bs4/dspdet3d_scannet-3d-22class.py'
DEFAULT_GAUSSIAN_CKPT = REPO_ROOT / 'work_dirs/scannet_gaussian_final_1gpu_bs4/latest.pth'
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize DSPDet3D baseline and Gaussian pruning.')
    parser.add_argument('--baseline-config', default=str(DEFAULT_BASELINE_CONFIG))
    parser.add_argument('--baseline-ckpt', default=str(DEFAULT_BASELINE_CKPT))
    parser.add_argument('--gaussian-config', default=str(DEFAULT_GAUSSIAN_CONFIG))
    parser.add_argument('--gaussian-ckpt', default=str(DEFAULT_GAUSSIAN_CKPT))
    parser.add_argument('--scene-idx', type=int, default=0)
    parser.add_argument('--score-thr', type=float, default=0.01)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', default=str(REPO_ROOT / 'work_dirs/pruning_vis'))
    parser.add_argument('--max-points-per-ply', type=int, default=150000)
    return parser.parse_args()


def convert_syncbn(cfg):
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == 'norm_cfg' and isinstance(value, dict) and 'type' in value:
                value['type'] = value['type'].replace('naiveSyncBN', 'BN')
            else:
                convert_syncbn(value)
    elif isinstance(cfg, (list, tuple)):
        for value in cfg:
            convert_syncbn(value)


def ensure_test_mode(cfg):
    if isinstance(cfg.data.test, list):
        for dataset_cfg in cfg.data.test:
            dataset_cfg.test_mode = True
    else:
        cfg.data.test.test_mode = True


def load_model_and_dataset(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    ensure_test_mode(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    convert_syncbn(cfg.model)

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    meta = checkpoint.get('meta', {})
    model.CLASSES = meta.get('CLASSES', getattr(dataset, 'CLASSES', cfg.get('class_names', None)))
    model.cfg = cfg
    model.to(device)
    model.eval()
    return cfg, dataset, model


def find_tensor(value):
    if torch.is_tensor(value):
        return value
    if hasattr(value, 'tensor') and torch.is_tensor(value.tensor):
        return value.tensor
    if hasattr(value, 'data'):
        return find_tensor(value.data)
    if isinstance(value, dict):
        for item in value.values():
            try:
                return find_tensor(item)
            except TypeError:
                pass
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return find_tensor(item)
            except TypeError:
                pass
    raise TypeError(f'No tensor found in {type(value)}')


def extract_points(sample):
    points = find_tensor(sample['points'])
    return points.detach().cpu().numpy()[:, :3]


def sample_to_batch(sample, device):
    data = collate([sample], samples_per_gpu=1)
    device_obj = torch.device(device)
    if device_obj.type == 'cuda':
        device_index = device_obj.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        data = scatter(data, [device_index])[0]
    else:
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    return data


def scene_name_from_dataset(dataset, scene_idx):
    info = dataset.data_infos[scene_idx]
    point_cloud = info.get('point_cloud', {})
    for key in ('lidar_idx', 'scene_idx', 'sample_idx'):
        if key in point_cloud:
            return str(point_cloud[key])
        if key in info:
            return str(info[key])
    pts_path = info.get('pts_path') or info.get('pts_filename') or f'scene_{scene_idx:06d}'
    return Path(pts_path).stem


def prune_mask_from_baseline(head, x, scores):
    with torch.no_grad():
        keep_score = 1 - scores.detach().sigmoid().reshape(-1)
        prune_mask = torch.zeros((len(keep_score),), dtype=torch.bool, device=keep_score.device)
        for permutation in x.decomposition_permutations:
            score = keep_score[permutation]
            mask = score > head.prune_threshold
            prune_mask[permutation[mask]] = True
    return keep_score, prune_mask


def sparse_coords_np(head, x):
    return (x.coordinates[:, 1:].detach().float().cpu().numpy() * head.voxel_size)


@contextlib.contextmanager
def capture_pruning(head):
    trace = []
    original_prune_inference = head._prune_inference
    original_prune_by_gaussian = head._prune_by_gaussian

    def traced_prune_inference(x, scores):
        keep_score, prune_mask = prune_mask_from_baseline(head, x, scores)
        trace.append(dict(
            kind='baseline',
            coords=sparse_coords_np(head, x),
            mask=prune_mask.detach().cpu().numpy(),
            score=keep_score.detach().cpu().numpy()))
        return original_prune_inference(x, scores)

    def traced_prune_by_gaussian(x, keep_prob):
        prune_mask = head._make_gaussian_prune_mask(keep_prob, x)
        trace.append(dict(
            kind='gaussian',
            coords=sparse_coords_np(head, x),
            mask=prune_mask.detach().cpu().numpy(),
            score=keep_prob.detach().cpu().numpy()))
        if prune_mask.sum() == 0:
            return x
        return head.pruning(x, prune_mask)

    head._prune_inference = traced_prune_inference
    head._prune_by_gaussian = traced_prune_by_gaussian
    try:
        yield trace
    finally:
        head._prune_inference = original_prune_inference
        head._prune_by_gaussian = original_prune_by_gaussian


def run_inference(model, dataset, scene_idx, device):
    sample = dataset[scene_idx]
    points = extract_points(sample)
    data = sample_to_batch(sample, device)
    with torch.no_grad(), capture_pruning(model.head) as trace:
        result = model(return_loss=False, rescale=True, **data)[0]
    assign_trace_levels(trace)
    return result, trace, points


def assign_trace_levels(trace):
    n_levels = len(trace)
    for idx, item in enumerate(trace):
        item['level_name'] = f'level_{n_levels - idx + 1}'


def downsample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    ids = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[ids]


def write_ply(points, colors, out_path):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(out_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for point, color in zip(points, colors):
            f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n')


def write_points_obj(points, out_path):
    with open(out_path, 'w') as f:
        for point in points:
            f.write(f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 160 160 160\n')


def write_colored_points(points, color, out_path, base_points=None, max_base_points=150000):
    point_parts = []
    color_parts = []
    if base_points is not None:
        base_points = downsample_points(base_points, max_base_points)
        point_parts.append(base_points)
        color_parts.append(np.full((len(base_points), 3), 120, dtype=np.uint8))
    if len(points) > 0:
        point_parts.append(points)
        color_parts.append(np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)))
    if point_parts:
        all_points = np.concatenate(point_parts, axis=0)
        all_colors = np.concatenate(color_parts, axis=0)
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)
        all_colors = np.zeros((0, 3), dtype=np.uint8)
    write_ply(all_points, all_colors, out_path)


def rounded_keys(points):
    rounded = np.round(points, 4)
    return [tuple(row.tolist()) for row in rounded]


def write_compare_ply(base_points, baseline_points, gaussian_points, out_path, max_base_points):
    base_points = downsample_points(base_points, max_base_points)
    point_parts = [base_points]
    color_parts = [np.full((len(base_points), 3), 120, dtype=np.uint8)]

    baseline_keys = rounded_keys(baseline_points)
    gaussian_keys = set(rounded_keys(gaussian_points))
    baseline_only = []
    common = []
    for point, key in zip(baseline_points, baseline_keys):
        if key in gaussian_keys:
            common.append(point)
        else:
            baseline_only.append(point)

    baseline_key_set = set(baseline_keys)
    gaussian_only = [point for point, key in zip(gaussian_points, rounded_keys(gaussian_points))
                     if key not in baseline_key_set]

    for points, color in ((np.asarray(baseline_only, dtype=np.float32), (40, 120, 255)),
                          (np.asarray(gaussian_only, dtype=np.float32), (255, 70, 70)),
                          (np.asarray(common, dtype=np.float32), (255, 255, 255))):
        if len(points) > 0:
            point_parts.append(points)
            color_parts.append(np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)))

    write_ply(np.concatenate(point_parts, axis=0), np.concatenate(color_parts, axis=0), out_path)


def boxes_to_corners(boxes_3d, keep):
    boxes = boxes_3d[keep]
    if len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    corners = boxes.corners
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()
    return corners.astype(np.float32)


def write_box_obj(corners, out_path, color):
    mtl_path = out_path.with_suffix('.mtl')
    material_name = out_path.stem
    with open(mtl_path, 'w') as f:
        rgb = np.asarray(color, dtype=np.float32) / 255.0
        f.write(f'newmtl {material_name}\n')
        f.write(f'Ka {rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f}\n')
        f.write(f'Kd {rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f}\n')

    with open(out_path, 'w') as f:
        f.write(f'mtllib {mtl_path.name}\n')
        f.write(f'usemtl {material_name}\n')
        vertex_offset = 1
        for box in corners:
            for vertex in box:
                f.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
            for start, end in BOX_EDGES:
                f.write(f'l {vertex_offset + start} {vertex_offset + end}\n')
            vertex_offset += 8


def save_predictions(result, out_dir, score_thr, classes, color, prefix='pred', prediction_name='predictions'):
    boxes_3d = result['boxes_3d']
    scores = result['scores_3d']
    labels = result['labels_3d']
    if torch.is_tensor(scores):
        keep = scores.detach().cpu() > score_thr
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
    else:
        keep = np.asarray(scores) > score_thr
        scores_np = np.asarray(scores)
        labels_np = np.asarray(labels)
    corners = boxes_to_corners(boxes_3d, keep)
    write_box_obj(corners, out_dir / f'{prefix}_boxes.obj', color)

    keep_np = keep.numpy() if torch.is_tensor(keep) else keep
    with open(out_dir / f'{prediction_name}.txt', 'w') as f:
        for score, label in zip(scores_np[keep_np], labels_np[keep_np]):
            name = classes[int(label)] if classes is not None and int(label) < len(classes) else str(int(label))
            f.write(f'{name} {float(score):.6f}\n')


def save_trace(name, trace, base_points, out_dir, max_points_per_ply):
    mmcv.mkdir_or_exist(str(out_dir))
    color = (40, 120, 255) if name == 'baseline' else (255, 70, 70)
    with open(out_dir / 'pruning_summary.txt', 'w') as f:
        f.write(f'model: {name}\n')
        for item in trace:
            mask = item['mask'].astype(bool)
            score = item['score']
            kept = int(mask.sum())
            total = int(len(mask))
            ratio = kept / max(total, 1)
            f.write(
                f"{item['level_name']}: kind={item['kind']} kept={kept} total={total} "
                f"ratio={ratio:.6f} score_min={score.min():.6f} score_max={score.max():.6f} "
                f"score_mean={score.mean():.6f}\n")
            kept_points = item['coords'][mask]
            write_colored_points(
                kept_points, color, out_dir / f"{item['level_name']}_pruning.ply",
                base_points=base_points, max_base_points=max_points_per_ply)


def save_compare_trace(baseline_trace, gaussian_trace, base_points, out_dir, max_points_per_ply):
    mmcv.mkdir_or_exist(str(out_dir))
    trace_by_level = {item['level_name']: item for item in baseline_trace}
    for gaussian_item in gaussian_trace:
        level_name = gaussian_item['level_name']
        if level_name not in trace_by_level:
            continue
        baseline_item = trace_by_level[level_name]
        baseline_points = baseline_item['coords'][baseline_item['mask'].astype(bool)]
        gaussian_points = gaussian_item['coords'][gaussian_item['mask'].astype(bool)]
        write_compare_ply(
            base_points, baseline_points, gaussian_points,
            out_dir / f'{level_name}_baseline_vs_gaussian.ply',
            max_points_per_ply)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    _, baseline_dataset, baseline_model = load_model_and_dataset(
        args.baseline_config, args.baseline_ckpt, device)
    _, gaussian_dataset, gaussian_model = load_model_and_dataset(
        args.gaussian_config, args.gaussian_ckpt, device)

    if args.scene_idx < 0 or args.scene_idx >= len(baseline_dataset):
        raise IndexError(f'--scene-idx must be in [0, {len(baseline_dataset) - 1}]')

    scene_name = scene_name_from_dataset(baseline_dataset, args.scene_idx)
    out_root = Path(args.out_dir) / f'{args.scene_idx:04d}_{scene_name}'
    baseline_dir = out_root / 'baseline'
    gaussian_dir = out_root / 'gaussian'
    compare_dir = out_root / 'compare'
    for path in (baseline_dir, gaussian_dir, compare_dir):
        mmcv.mkdir_or_exist(str(path))

    baseline_result, baseline_trace, points = run_inference(
        baseline_model, baseline_dataset, args.scene_idx, device)
    gaussian_result, gaussian_trace, _ = run_inference(
        gaussian_model, gaussian_dataset, args.scene_idx, device)

    write_points_obj(points, baseline_dir / 'points.obj')
    write_points_obj(points, gaussian_dir / 'points.obj')
    write_colored_points(
        np.zeros((0, 3), dtype=np.float32), (120, 120, 120), compare_dir / 'points.ply',
        base_points=points, max_base_points=args.max_points_per_ply)

    save_predictions(baseline_result, baseline_dir, args.score_thr, baseline_model.CLASSES, (40, 120, 255))
    save_predictions(gaussian_result, gaussian_dir, args.score_thr, gaussian_model.CLASSES, (255, 70, 70))
    save_predictions(
        baseline_result, compare_dir, args.score_thr, baseline_model.CLASSES,
        (40, 120, 255), prefix='baseline_pred', prediction_name='baseline_predictions')
    save_predictions(
        gaussian_result, compare_dir, args.score_thr, gaussian_model.CLASSES,
        (255, 70, 70), prefix='gaussian_pred', prediction_name='gaussian_predictions')

    save_trace('baseline', baseline_trace, points, baseline_dir, args.max_points_per_ply)
    save_trace('gaussian', gaussian_trace, points, gaussian_dir, args.max_points_per_ply)
    save_compare_trace(baseline_trace, gaussian_trace, points, compare_dir, args.max_points_per_ply)

    with open(out_root / 'summary.txt', 'w') as f:
        f.write(f'scene_idx: {args.scene_idx}\n')
        f.write(f'scene_name: {scene_name}\n')
        f.write(f'baseline_config: {args.baseline_config}\n')
        f.write(f'baseline_ckpt: {args.baseline_ckpt}\n')
        f.write(f'baseline_gaussian_enabled: {baseline_model.head.gaussian_pruning_enabled}\n')
        f.write(f'gaussian_config: {args.gaussian_config}\n')
        f.write(f'gaussian_ckpt: {args.gaussian_ckpt}\n')
        f.write(f'gaussian_gaussian_enabled: {gaussian_model.head.gaussian_pruning_enabled}\n')
        f.write(f'score_thr: {args.score_thr}\n')
        f.write('colors: baseline=blue, gaussian=red, common=white, scene_points=gray\n')

    print(f'Wrote pruning visualization to {out_root}')


if __name__ == '__main__':
    main()
