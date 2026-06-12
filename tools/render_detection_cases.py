import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np


CLASSES = (
    'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup',
    'curtain', 'desk', 'door', 'dresser', 'keyboard', 'lamp', 'laptop',
    'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
    'wardrobe')
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render RGB ScanNet point clouds with GT/baseline/fixsmall boxes.')
    parser.add_argument('--data-root', default='data/ScanNet-md40/mmdet_scannet')
    parser.add_argument('--ann-file', default='data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl')
    parser.add_argument('--baseline-pkl', required=True)
    parser.add_argument('--fixsmall-pkl', required=True)
    parser.add_argument('--scene-idx', nargs='+', type=int, required=True)
    parser.add_argument('--out-dir', default='work_dirs/pretty_case_vis')
    parser.add_argument('--score-thr', type=float, default=0.01)
    parser.add_argument('--min-delta', type=float, default=0.20)
    parser.add_argument('--min-fix-iou', type=float, default=0.25)
    parser.add_argument('--max-points-bev', type=int, default=120000)
    parser.add_argument('--max-points-3d', type=int, default=60000)
    parser.add_argument('--azim', type=float, default=-55)
    parser.add_argument('--elev', type=float, default=35)
    parser.add_argument('--dpi', type=int, default=260)
    parser.add_argument('--show-all-gt', action='store_true')
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def result_boxes(result):
    boxes = result['boxes_3d']
    tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
    return as_numpy(tensor).astype(np.float32)


def load_points(info, data_root, max_points):
    points = np.fromfile(data_root / info['pts_path'], dtype=np.float32).reshape(-1, 6)
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    axis_align_matrix = info['annos'].get('axis_align_matrix')
    if axis_align_matrix is not None:
        xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
        xyz = (xyz_h @ axis_align_matrix.T)[:, :3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    if max_points > 0 and len(xyz) > max_points:
        ids = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[ids]
        rgb = rgb[ids]
    return xyz, rgb


def box_minmax(box):
    x, y, z, dx, dy, dz = box[:6]
    return np.array([x - dx / 2, y - dy / 2, z - dz / 2], dtype=np.float32), \
        np.array([x + dx / 2, y + dy / 2, z + dz / 2], dtype=np.float32)


def box_corners(box):
    mn, mx = box_minmax(box)
    xs = [mn[0], mx[0]]
    ys = [mn[1], mx[1]]
    zs = [mn[2], mx[2]]
    return np.array([
        [xs[0], ys[0], zs[0]], [xs[1], ys[0], zs[0]],
        [xs[1], ys[1], zs[0]], [xs[0], ys[1], zs[0]],
        [xs[0], ys[0], zs[1]], [xs[1], ys[0], zs[1]],
        [xs[1], ys[1], zs[1]], [xs[0], ys[1], zs[1]],
    ], dtype=np.float32)


def box_center(box):
    mn, mx = box_minmax(box)
    return (mn + mx) / 2


def box_iou_3d(box1, box2):
    mn1, mx1 = box_minmax(box1)
    mn2, mx2 = box_minmax(box2)
    inter = np.maximum(0, np.minimum(mx1, mx2) - np.maximum(mn1, mn2)).prod()
    vol1 = np.maximum(0, mx1 - mn1).prod()
    vol2 = np.maximum(0, mx2 - mn2).prod()
    return float(inter / (vol1 + vol2 - inter + 1e-9))


def best_same_class(gt_box, gt_label, pred_boxes, pred_labels, pred_scores, score_thr):
    best = dict(iou=0.0, score=0.0, index=-1, box=None)
    for idx, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        if int(label) != int(gt_label) or float(score) < score_thr:
            continue
        iou = box_iou_3d(gt_box, box)
        if iou > best['iou']:
            best = dict(iou=iou, score=float(score), index=idx, box=box)
    return best


def select_wins(info, baseline_result, fixsmall_result, args):
    ann = info['annos']
    gt_boxes = np.asarray(ann['gt_boxes_upright_depth'], dtype=np.float32)
    gt_labels = np.asarray(ann['class'], dtype=np.int64)
    gt_names = list(ann['name'])

    baseline_boxes = result_boxes(baseline_result)
    baseline_labels = as_numpy(baseline_result['labels_3d']).astype(np.int64)
    baseline_scores = as_numpy(baseline_result['scores_3d']).astype(np.float32)
    fixsmall_boxes = result_boxes(fixsmall_result)
    fixsmall_labels = as_numpy(fixsmall_result['labels_3d']).astype(np.int64)
    fixsmall_scores = as_numpy(fixsmall_result['scores_3d']).astype(np.float32)

    wins = []
    all_matches = []
    for gt_idx, (gt_box, gt_label, gt_name) in enumerate(zip(gt_boxes, gt_labels, gt_names)):
        baseline = best_same_class(
            gt_box, gt_label, baseline_boxes, baseline_labels, baseline_scores, args.score_thr)
        fixsmall = best_same_class(
            gt_box, gt_label, fixsmall_boxes, fixsmall_labels, fixsmall_scores, args.score_thr)
        delta = fixsmall['iou'] - baseline['iou']
        match = dict(
            gt_idx=gt_idx, name=gt_name, gt_box=gt_box, baseline=baseline,
            fixsmall=fixsmall, delta=delta)
        all_matches.append(match)
        if delta >= args.min_delta and fixsmall['iou'] >= args.min_fix_iou:
            wins.append(match)

    if not wins:
        wins = sorted(
            [m for m in all_matches if m['fixsmall']['iou'] > m['baseline']['iou']],
            key=lambda m: m['delta'], reverse=True)[:3]
    return wins, all_matches


def draw_boxes_bev(ax, boxes, color, linewidth=1.6, alpha=0.95):
    for box in boxes:
        corners = box_corners(box)
        for start, end in BOX_EDGES[:4]:
            ax.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                color=color, linewidth=linewidth, alpha=alpha)


def draw_boxes_3d(ax, boxes, color, linewidth=1.2, alpha=0.95):
    for box in boxes:
        corners = box_corners(box)
        for start, end in BOX_EDGES:
            ax.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]],
                color=color, linewidth=linewidth, alpha=alpha)


def set_equal_3d(ax, points, boxes):
    parts = [points]
    for group in boxes:
        for box in group:
            parts.append(box_corners(box))
    all_points = np.concatenate(parts, axis=0)
    mn = all_points.min(axis=0)
    mx = all_points.max(axis=0)
    center = (mn + mx) / 2
    radius = (mx - mn).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0, center[2] - radius * 0.35), center[2] + radius * 0.75)


def render_scene(scene_idx, info, baseline_result, fixsmall_result, args, out_root):
    scene_name = str(info['point_cloud'].get('lidar_idx', scene_idx))
    out_dir = out_root / f'{scene_idx:04d}_{scene_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    wins, all_matches = select_wins(info, baseline_result, fixsmall_result, args)
    points_bev, colors_bev = load_points(info, Path(args.data_root), args.max_points_bev)
    points_3d, colors_3d = load_points(info, Path(args.data_root), args.max_points_3d)

    gt_boxes_all = np.asarray(info['annos']['gt_boxes_upright_depth'], dtype=np.float32)
    gt_boxes = [m['gt_box'] for m in wins]
    baseline_boxes = [m['baseline']['box'] for m in wins if m['baseline']['box'] is not None]
    fixsmall_boxes = [m['fixsmall']['box'] for m in wins if m['fixsmall']['box'] is not None]

    fig, ax = plt.subplots(figsize=(11, 9), facecolor='white')
    ax.scatter(
        points_bev[:, 0], points_bev[:, 1], c=colors_bev, s=0.18,
        alpha=0.55, linewidths=0, rasterized=True)
    if args.show_all_gt:
        draw_boxes_bev(ax, gt_boxes_all, '#8fd18f', linewidth=0.8, alpha=0.35)
    draw_boxes_bev(ax, gt_boxes, '#00aa00', linewidth=2.3, alpha=0.98)
    draw_boxes_bev(ax, baseline_boxes, '#1f77b4', linewidth=1.8, alpha=0.95)
    draw_boxes_bev(ax, fixsmall_boxes, '#d62728', linewidth=1.8, alpha=0.95)
    for match in wins:
        center = box_center(match['gt_box'])
        ax.text(
            center[0], center[1],
            f"{match['name']}\n{match['baseline']['iou']:.2f}->{match['fixsmall']['iou']:.2f}",
            fontsize=8, color='black', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.65))
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(
        f'{scene_name} BEV selected wins\nGT green, baseline blue, fixsmall red',
        fontsize=13)
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#00aa00', lw=2, label='GT'),
        plt.Line2D([0], [0], color='#1f77b4', lw=2, label='baseline best'),
        plt.Line2D([0], [0], color='#d62728', lw=2, label='fixsmall best'),
    ], loc='upper right', framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / 'pretty_bev.png', dpi=args.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors_3d,
        s=0.18, alpha=0.45, linewidths=0, rasterized=True)
    draw_boxes_3d(ax, gt_boxes, '#00aa00', linewidth=1.8, alpha=0.98)
    draw_boxes_3d(ax, baseline_boxes, '#1f77b4', linewidth=1.2, alpha=0.95)
    draw_boxes_3d(ax, fixsmall_boxes, '#d62728', linewidth=1.2, alpha=0.95)
    set_equal_3d(ax, points_3d, [gt_boxes, baseline_boxes, fixsmall_boxes])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_title(
        f'{scene_name} 3D selected wins\nGT green, baseline blue, fixsmall red',
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / 'pretty_3d.png', dpi=args.dpi)
    plt.close(fig)

    with open(out_dir / 'pretty_summary.txt', 'w') as f:
        f.write(f'scene_idx: {scene_idx}\n')
        f.write(f'scene_name: {scene_name}\n')
        f.write(f'all_gt_classes: {list(info["annos"]["name"])}\n')
        f.write(f'gt_class_counts: {dict(Counter(info["annos"]["name"]))}\n')
        f.write(f'num_selected_wins: {len(wins)}\n')
        for match in sorted(wins, key=lambda m: m['delta'], reverse=True):
            f.write(
                f"WIN gt#{match['gt_idx']:02d} {match['name']} "
                f"base_iou={match['baseline']['iou']:.3f} "
                f"fix_iou={match['fixsmall']['iou']:.3f} "
                f"delta={match['delta']:+.3f} "
                f"base_score={match['baseline']['score']:.3f} "
                f"fix_score={match['fixsmall']['score']:.3f}\n")
    return out_dir


def main():
    args = parse_args()
    infos = mmcv.load(args.ann_file)
    baseline_results = mmcv.load(args.baseline_pkl)
    fixsmall_results = mmcv.load(args.fixsmall_pkl)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for scene_idx in args.scene_idx:
        out_dir = render_scene(
            scene_idx, infos[scene_idx], baseline_results[scene_idx],
            fixsmall_results[scene_idx], args, out_root)
        print(f'Wrote {out_dir / "pretty_bev.png"}')
        print(f'Wrote {out_dir / "pretty_3d.png"}')


if __name__ == '__main__':
    main()
