import argparse
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
    parser = argparse.ArgumentParser(description='Render GT/TP/FN/FP detection cases.')
    parser.add_argument('--data-root', default='data/ScanNet-md40/mmdet_scannet')
    parser.add_argument('--ann-file', default='data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl')
    parser.add_argument('--baseline-pkl', required=True)
    parser.add_argument('--fixsmall-pkl', required=True)
    parser.add_argument('--scene-idx', nargs='+', type=int, required=True)
    parser.add_argument('--out-dir', default='work_dirs/tp_fp_fn_case_vis')
    parser.add_argument('--score-thr', type=float, default=0.10)
    parser.add_argument('--iou-thr', type=float, default=0.25)
    parser.add_argument('--fp-score-thr', type=float, default=None)
    parser.add_argument('--max-fp', type=int, default=40)
    parser.add_argument('--max-points', type=int, default=130000)
    parser.add_argument('--dpi', type=int, default=260)
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
    rgb = np.clip(rgb * 1.35, 0.0, 1.0)
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


def filter_predictions(result, score_thr):
    boxes = result_boxes(result)
    labels = as_numpy(result['labels_3d']).astype(np.int64)
    scores = as_numpy(result['scores_3d']).astype(np.float32)
    keep = np.where(scores >= score_thr)[0]
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes[keep], labels[keep], scores[keep], keep


def greedy_match(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thr):
    candidates = []
    for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if int(pred_label) != int(gt_label):
                continue
            iou = box_iou_3d(gt_box, pred_box)
            if iou >= iou_thr:
                candidates.append((iou, float(pred_score), pred_idx, gt_idx))
    candidates.sort(reverse=True)

    used_preds = set()
    used_gts = set()
    matches = []
    for iou, score, pred_idx, gt_idx in candidates:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matches.append(dict(pred_idx=pred_idx, gt_idx=gt_idx, iou=iou, score=score))
    fn = [idx for idx in range(len(gt_boxes)) if idx not in used_gts]
    fp = [idx for idx in range(len(pred_boxes)) if idx not in used_preds]
    return matches, fn, fp


def draw_boxes(ax, boxes, color, linewidth=1.4, alpha=0.9, linestyle='-'):
    for box in boxes:
        corners = box_corners(box)
        for start, end in BOX_EDGES[:4]:
            ax.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def annotate_gt(ax, gt_boxes, gt_names, indices, color):
    for idx in indices:
        center = box_center(gt_boxes[idx])
        ax.text(
            center[0], center[1], gt_names[idx], fontsize=7, color=color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.65))


def render_panel(scene_name, points, colors, gt_boxes, gt_names, pred_boxes, pred_labels,
                 pred_scores, matches, fn, fp, title, out_path, args):
    fp_score_thr = args.fp_score_thr if args.fp_score_thr is not None else args.score_thr
    fp = [idx for idx in fp if pred_scores[idx] >= fp_score_thr]
    fp = sorted(fp, key=lambda idx: pred_scores[idx], reverse=True)[:args.max_fp]
    tp_pred_indices = [m['pred_idx'] for m in matches]
    tp_gt_indices = [m['gt_idx'] for m in matches]

    fig, ax = plt.subplots(figsize=(11, 9), facecolor='white')
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.25, alpha=0.62, linewidths=0, rasterized=True)
    draw_boxes(ax, gt_boxes, '#00aa00', linewidth=2.1, alpha=0.95)
    draw_boxes(ax, pred_boxes[tp_pred_indices], '#1f77b4', linewidth=1.8, alpha=0.95)
    draw_boxes(ax, gt_boxes[fn], '#ff9900', linewidth=2.4, alpha=0.98, linestyle='--')
    draw_boxes(ax, pred_boxes[fp], '#d62728', linewidth=1.2, alpha=0.72)
    annotate_gt(ax, gt_boxes, gt_names, fn, '#cc6600')

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(
        f'{scene_name} {title}\nGT green, TP pred blue, FN orange dashed, FP red',
        fontsize=13)
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#00aa00', lw=2, label=f'GT {len(gt_boxes)}'),
        plt.Line2D([0], [0], color='#1f77b4', lw=2, label=f'TP {len(matches)}'),
        plt.Line2D([0], [0], color='#ff9900', lw=2, linestyle='--', label=f'FN {len(fn)}'),
        plt.Line2D([0], [0], color='#d62728', lw=2, label=f'FP {len(fp)}'),
    ], loc='upper right', framealpha=0.88)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def render_gt_only(scene_name, points, colors, gt_boxes, gt_names, out_path, args):
    fig, ax = plt.subplots(figsize=(11, 9), facecolor='white')
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.25, alpha=0.62, linewidths=0, rasterized=True)
    draw_boxes(ax, gt_boxes, '#00aa00', linewidth=2.1, alpha=0.95)
    annotate_gt(ax, gt_boxes, gt_names, range(len(gt_boxes)), '#006600')
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(f'{scene_name} GT only', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def render_scene(scene_idx, info, baseline_result, fixsmall_result, args, out_root):
    scene_name = str(info['point_cloud'].get('lidar_idx', scene_idx))
    out_dir = out_root / f'{scene_idx:04d}_{scene_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_boxes = np.asarray(info['annos']['gt_boxes_upright_depth'], dtype=np.float32)
    gt_labels = np.asarray(info['annos']['class'], dtype=np.int64)
    gt_names = list(info['annos']['name'])
    points, colors = load_points(info, Path(args.data_root), args.max_points)

    baseline_boxes, baseline_labels, baseline_scores, _ = filter_predictions(baseline_result, args.score_thr)
    fixsmall_boxes, fixsmall_labels, fixsmall_scores, _ = filter_predictions(fixsmall_result, args.score_thr)
    baseline_matches, baseline_fn, baseline_fp = greedy_match(
        gt_boxes, gt_labels, baseline_boxes, baseline_labels, baseline_scores, args.iou_thr)
    fixsmall_matches, fixsmall_fn, fixsmall_fp = greedy_match(
        gt_boxes, gt_labels, fixsmall_boxes, fixsmall_labels, fixsmall_scores, args.iou_thr)

    render_gt_only(scene_name, points, colors, gt_boxes, gt_names, out_dir / 'gt_only.png', args)
    render_panel(
        scene_name, points, colors, gt_boxes, gt_names, baseline_boxes,
        baseline_labels, baseline_scores, baseline_matches, baseline_fn, baseline_fp,
        'baseline', out_dir / 'baseline_tp_fn_fp.png', args)
    render_panel(
        scene_name, points, colors, gt_boxes, gt_names, fixsmall_boxes,
        fixsmall_labels, fixsmall_scores, fixsmall_matches, fixsmall_fn, fixsmall_fp,
        'GMM/fixsmall', out_dir / 'gmm_tp_fn_fp.png', args)

    with open(out_dir / 'tp_fp_fn_summary.txt', 'w') as f:
        f.write(f'scene_idx: {scene_idx}\n')
        f.write(f'scene_name: {scene_name}\n')
        f.write(f'score_thr: {args.score_thr}\n')
        f.write(f'iou_thr: {args.iou_thr}\n')
        f.write(f'gt: {list(gt_names)}\n')
        for name, matches, fn, fp, boxes, labels, scores in [
                ('baseline', baseline_matches, baseline_fn, baseline_fp, baseline_boxes, baseline_labels, baseline_scores),
                ('gmm', fixsmall_matches, fixsmall_fn, fixsmall_fp, fixsmall_boxes, fixsmall_labels, fixsmall_scores)]:
            fp_score_thr = args.fp_score_thr if args.fp_score_thr is not None else args.score_thr
            fp_kept = [idx for idx in fp if scores[idx] >= fp_score_thr]
            f.write(f'\n{name}: TP={len(matches)} FN={len(fn)} FP={len(fp_kept)} predictions_above_thr={len(boxes)}\n')
            for match in sorted(matches, key=lambda m: m['iou'], reverse=True):
                gt_idx = match['gt_idx']
                pred_idx = match['pred_idx']
                f.write(
                    f"  TP gt#{gt_idx:02d} {gt_names[gt_idx]} "
                    f"iou={match['iou']:.3f} score={match['score']:.3f}\n")
            for gt_idx in fn:
                f.write(f"  FN gt#{gt_idx:02d} {gt_names[gt_idx]}\n")
            for pred_idx in sorted(fp_kept, key=lambda idx: scores[idx], reverse=True)[:args.max_fp]:
                f.write(
                    f"  FP {CLASSES[int(labels[pred_idx])]} "
                    f"score={float(scores[pred_idx]):.3f}\n")
    return out_dir


def main():
    args = parse_args()
    infos = mmcv.load(args.ann_file)
    baseline_results = mmcv.load(args.baseline_pkl)
    fixsmall_results = mmcv.load(args.fixsmall_pkl)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for scene_idx in args.scene_idx:
        out_dir = render_scene(scene_idx, infos[scene_idx], baseline_results[scene_idx], fixsmall_results[scene_idx], args, out_root)
        print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
