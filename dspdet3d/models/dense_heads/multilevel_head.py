import numpy as np

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner


import pdb

@HEADS.register_module()
class DSPHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_prune_threshold,
                 assigner,
                 volume_threshold,
                 r,
                 assign_type='volume',
                 prune_threshold=0,
                 gaussian_pruning=None,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 keep_loss=dict(type='FocalLoss', reduction='mean', use_sigmoid=True),
                 train_cfg=None,
                 test_cfg=None):
        super(DSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.r = r
        self.prune_threshold = prune_threshold
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.keep_loss = build_loss(keep_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.current_epoch = 0
        self._init_gaussian_pruning(gaussian_pruning)
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
        self._freeze_inactive_pruning_heads()


    def _init_gaussian_pruning(self, gaussian_pruning):
        gaussian_pruning = gaussian_pruning or {}
        self.gaussian_pruning_enabled = gaussian_pruning.get('enabled', False)
        self.gmm_num_primitives = gaussian_pruning.get('num_primitives', 1)
        self.gmm_keep_threshold = gaussian_pruning.get('keep_threshold', self.prune_threshold)
        self.gmm_min_keep = gaussian_pruning.get('min_keep', 1)
        self.gmm_max_keep = gaussian_pruning.get('max_keep', self.pts_prune_threshold)
        self.gmm_warmup_epochs = gaussian_pruning.get('warmup_epochs', 0)
        self.gmm_chunk_size = gaussian_pruning.get('chunk_size', 512)
        self.gmm_knn_k = gaussian_pruning.get('knn_k', 8)
        self.gmm_neighbor_backend = gaussian_pruning.get('neighbor_backend', 'cdist')
        self.gmm_local_window_radius = gaussian_pruning.get('local_window_radius', 1)
        self.gmm_local_cell_size_scale = gaussian_pruning.get('local_cell_size_scale', 1.0)
        self.gmm_local_dense_max_cells = gaussian_pruning.get('local_dense_max_cells', 2000000)
        self.gmm_local_fallback = gaussian_pruning.get('local_fallback', 'none')
        self.gmm_local_fallback_radius = gaussian_pruning.get('local_fallback_radius', self.gmm_local_window_radius)
        self.gmm_train_gate_floor = gaussian_pruning.get('train_gate_floor', 0.05)
        self.gmm_scale_min = gaussian_pruning.get('scale_min', gaussian_pruning.get('sigma_min', 0.1))
        self.gmm_scale_max = gaussian_pruning.get('scale_max', gaussian_pruning.get('sigma_max', 2.0))
        self.gmm_volume_loss_weight = gaussian_pruning.get('volume_loss_weight', 0.005)
        self.gmm_sparsity_loss_weight = gaussian_pruning.get('opacity_sparsity_loss_weight', 0.01)
        self.gmm_loss_weight = gaussian_pruning.get('gmm_loss_weight', gaussian_pruning.get('loss_weight', 0.01))


    def _freeze_inactive_pruning_heads(self):
        inactive_heads = [self.keep_conv] if self.gaussian_pruning_enabled else [
            self.opacity_conv, self.scale_conv, self.rot_conv]
        for heads in inactive_heads:
            for parameter in heads.parameters():
                parameter.requires_grad = False


    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(in_channels[i + 1], 1, kernel_size=1, bias=True, dimension=3)
            for i in range(len(in_channels) - 1)
        ])
        self.opacity_conv = nn.ModuleList([
            ME.MinkowskiConvolution(
                in_channels[i + 1], self.gmm_num_primitives, kernel_size=1, bias=True, dimension=3)
            for i in range(len(in_channels) - 1)
        ])
        self.scale_conv = nn.ModuleList([
            ME.MinkowskiConvolution(
                in_channels[i + 1], self.gmm_num_primitives * 3, kernel_size=1, bias=True, dimension=3)
            for i in range(len(in_channels) - 1)
        ])
        self.rot_conv = nn.ModuleList([
            ME.MinkowskiConvolution(
                in_channels[i + 1], self.gmm_num_primitives * 4, kernel_size=1, bias=True, dimension=3)
            for i in range(len(in_channels) - 1)
        ])
        self.pruning = ME.MinkowskiPruning()

        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            # if i < len(in_channels) - 1:
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                        f'out_block_{i}',
                        self.make_block(in_channels[i], out_channels))
        # ######only train keep_head
        # for name, param in self.named_parameters():
        #     if "keep_conv" not in name:
        #         param.requires_grad=False


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)
            nn.init.normal_(self.opacity_conv[i].kernel, std=.01)
            nn.init.constant_(self.opacity_conv[i].bias, bias_init_with_prob(.8))
            nn.init.normal_(self.scale_conv[i].kernel, std=.01)
            nn.init.constant_(self.scale_conv[i].bias, 0)
            nn.init.normal_(self.rot_conv[i].kernel, std=.01)
            nn.init.constant_(self.rot_conv[i].bias, 0)
            with torch.no_grad():
                self.rot_conv[i].bias.view(-1)[0::4].fill_(1)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) \
                and ('keep_conv' not in n) and ('opacity_conv' not in n) \
                and ('scale_conv' not in n) and ('rot_conv' not in n) \
                and ('loss' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)


    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features
        prune_training = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)
        return bbox_preds, cls_preds, points, prune_training


    def _sparse_like(self, x, features):
        return ME.SparseTensor(
            features=features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)


    def _gmm_level_spacing(self, transition_idx):
        return self.voxel_size * (2 ** (transition_idx + 2))


    @staticmethod
    def _quaternion_to_matrix(rotation):
        rot = F.normalize(rotation, dim=-1, eps=1e-6)
        r, x, y, z = rot.unbind(dim=-1)
        matrix = torch.stack([
            1 - 2 * (y ** 2 + z ** 2),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x ** 2 + z ** 2),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x ** 2 + y ** 2)
        ], dim=-1)
        return matrix.reshape(rotation.shape[:-1] + (3, 3))


    def _compute_covariance_3d(self, scaling, rotation):
        scale = torch.exp(scaling)
        rot = self._quaternion_to_matrix(rotation)
        scale_matrix = torch.diag_embed(scale)
        transform = rot @ scale_matrix
        sigma = transform @ transform.transpose(-1, -2)
        eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype).view(1, 3, 3)
        sigma_inv = torch.inverse(sigma + 1e-6 * eye)
        return sigma, sigma_inv


    def _decode_gmm_params(self, x, transition_idx):
        n_primitives = self.gmm_num_primitives
        opacity = torch.sigmoid(self.opacity_conv[transition_idx](x).features)
        scale_logits = self.scale_conv[transition_idx](x).features.reshape(-1, n_primitives, 3)
        rotation = self.rot_conv[transition_idx](x).features.reshape(-1, n_primitives, 4)
        scale = self.gmm_scale_min + (self.gmm_scale_max - self.gmm_scale_min) * torch.sigmoid(scale_logits)
        scale = scale * self._gmm_level_spacing(transition_idx)
        return dict(
            coords=x.coordinates[:, 1:].float() * self.voxel_size,
            permutations=x.decomposition_permutations,
            opacity=opacity.reshape(-1, n_primitives),
            scale=scale,
            rotation=self._quaternion_to_matrix(rotation),
            spacing=x.features.new_tensor(self._gmm_level_spacing(transition_idx)))


    def _gmm_regularizers(self, gmm_params):
        opacity = gmm_params['opacity']
        scale = gmm_params['scale']
        spacing = gmm_params['spacing']
        volume = torch.prod(scale / spacing.clamp_min(1e-6), dim=-1).mean()
        opacity_entropy = -(opacity * torch.log(opacity + 1e-6) +
                            (1 - opacity) * torch.log(1 - opacity + 1e-6)).mean()
        return volume, opacity_entropy


    def _score_gmm_neighbors(self, target_coords, nn_coords, nn_opacity,
                             nn_scale, nn_rotation, valid_mask=None):
        delta = target_coords[:, None, None, :] - nn_coords[:, :, None, :]
        delta = delta.expand(-1, -1, self.gmm_num_primitives, -1)
        delta_local = torch.matmul(
            nn_rotation.transpose(-1, -2), delta.unsqueeze(-1)).squeeze(-1)
        dist = torch.sum((delta_local / nn_scale.clamp_min(1e-6)) ** 2, dim=-1)
        scores = nn_opacity * torch.exp(-0.5 * dist)
        if valid_mask is not None:
            scores = scores * valid_mask.unsqueeze(-1)
        return scores.reshape(len(target_coords), -1).max(dim=1).values


    def _evaluate_gmm_field_cdist(self, target_x, gmm_params):
        keep_prob = target_x.features.new_zeros((len(target_x.features),))
        opacity = gmm_params['opacity']
        scale = gmm_params['scale']
        rotation = gmm_params['rotation']
        source_coords = gmm_params['coords']
        chunk_size = max(int(self.gmm_chunk_size), 1)

        for source_perm, target_perm in zip(gmm_params['permutations'], target_x.decomposition_permutations):
            if len(source_perm) == 0 or len(target_perm) == 0:
                continue
            scene_source_coords = source_coords[source_perm]
            scene_opacity = opacity[source_perm]
            scene_scale = scale[source_perm]
            scene_rotation = rotation[source_perm]
            scene_target_coords = target_x.coordinates[target_perm][:, 1:].float() * self.voxel_size
            k = min(int(self.gmm_knn_k), len(scene_source_coords))
            for start in range(0, len(scene_target_coords), chunk_size):
                end = min(start + chunk_size, len(scene_target_coords))
                target_chunk = scene_target_coords[start:end]
                nn_ids = torch.cdist(target_chunk, scene_source_coords).topk(
                    k, largest=False, sorted=False).indices
                keep_prob[target_perm[start:end]] = self._score_gmm_neighbors(
                    target_chunk,
                    scene_source_coords[nn_ids],
                    scene_opacity[nn_ids],
                    scene_scale[nn_ids],
                    scene_rotation[nn_ids])
        return keep_prob


    def _gmm_local_offsets(self, radius, device):
        radius = max(int(radius), 0)
        values = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
        offsets = torch.meshgrid(values, values, values, indexing='ij')
        return torch.stack([offset.reshape(-1) for offset in offsets], dim=1)


    @staticmethod
    def _linearize_gmm_cells(cells, min_cell, strides):
        return ((cells - min_cell) * strides).sum(dim=-1)


    def _build_gmm_cell_lookup(self, source_cells, min_cell, grid_shape):
        strides = torch.stack((grid_shape[1] * grid_shape[2],
                               grid_shape[2],
                               grid_shape.new_tensor(1)))
        source_keys = self._linearize_gmm_cells(source_cells, min_cell, strides)
        total_cells = int((grid_shape[0] * grid_shape[1] * grid_shape[2]).item())
        sorted_keys, order = torch.sort(source_keys)
        unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        cell_ids = torch.arange(len(unique_keys), device=source_keys.device)
        max_sources_per_cell = int(counts.max().item()) if len(counts) > 0 else 0
        cell_source_ids = order.new_full((len(unique_keys), max_sources_per_cell), -1)

        if len(order) > 0:
            cell_starts = counts.cumsum(dim=0) - counts
            slot_ids = torch.arange(len(order), device=order.device) - torch.repeat_interleave(
                cell_starts, counts)
            repeated_cell_ids = torch.repeat_interleave(cell_ids, counts)
            cell_source_ids[repeated_cell_ids, slot_ids] = order

        if total_cells <= int(self.gmm_local_dense_max_cells):
            dense_lookup = source_keys.new_full((total_cells,), -1)
            dense_lookup[unique_keys] = cell_ids
            return dict(
                mode='dense',
                dense_lookup=dense_lookup,
                cell_source_ids=cell_source_ids,
                strides=strides)

        return dict(
            mode='sorted',
            sorted_keys=unique_keys,
            cell_source_ids=cell_source_ids,
            strides=strides)


    def _gather_gmm_local_candidates(self, target_cells, offsets, min_cell,
                                     grid_shape, lookup):
        candidate_cells = target_cells[:, None, :] + offsets[None, :, :]
        valid_cells = ((candidate_cells >= min_cell) &
                       (candidate_cells < min_cell + grid_shape)).all(dim=-1)
        clamped_cells = torch.max(torch.min(candidate_cells, min_cell + grid_shape - 1), min_cell)
        candidate_keys = self._linearize_gmm_cells(clamped_cells, min_cell, lookup['strides'])

        if lookup['mode'] == 'dense':
            candidate_cell_ids = lookup['dense_lookup'][candidate_keys]
        else:
            flat_keys = candidate_keys.reshape(-1)
            positions = torch.searchsorted(lookup['sorted_keys'], flat_keys)
            in_range = positions < len(lookup['sorted_keys'])
            safe_positions = positions.clamp(max=max(len(lookup['sorted_keys']) - 1, 0))
            found = in_range & (lookup['sorted_keys'][safe_positions] == flat_keys)
            flat_ids = flat_keys.new_full((len(flat_keys),), -1)
            flat_ids[found] = safe_positions[found]
            candidate_cell_ids = flat_ids.reshape(candidate_keys.shape)

        valid_cells = valid_cells & (candidate_cell_ids >= 0)
        if lookup['cell_source_ids'].numel() == 0:
            return candidate_cell_ids.new_full(candidate_cell_ids.shape, -1)

        safe_cell_ids = candidate_cell_ids.clamp_min(0)
        candidate_ids = lookup['cell_source_ids'][safe_cell_ids]
        candidate_ids = torch.where(
            valid_cells.unsqueeze(-1),
            candidate_ids,
            candidate_ids.new_full((), -1))
        return candidate_ids.reshape(len(target_cells), -1)


    def _score_gmm_local_candidates(self, target_chunk, target_cells, scene_source_coords,
                                    scene_opacity, scene_scale, scene_rotation,
                                    offsets, min_cell, grid_shape, lookup):
        local_ids = self._gather_gmm_local_candidates(
            target_cells, offsets, min_cell, grid_shape, lookup)
        valid_mask = local_ids >= 0
        if not valid_mask.any():
            return target_chunk.new_zeros((len(target_chunk),)), valid_mask.any(dim=1)

        safe_ids = local_ids.clamp_min(0)
        candidate_coords = scene_source_coords[safe_ids]
        candidate_distances = torch.sum((candidate_coords - target_chunk[:, None, :]) ** 2, dim=-1)
        candidate_distances = candidate_distances.masked_fill(~valid_mask, float('inf'))

        k = min(int(self.gmm_knn_k), candidate_distances.shape[1])
        if k > 0 and candidate_distances.shape[1] > k:
            top_distances, top_ids = torch.topk(candidate_distances, k, largest=False, sorted=False)
            local_ids = torch.gather(local_ids, 1, top_ids)
            valid_mask = torch.gather(valid_mask, 1, top_ids) & torch.isfinite(top_distances)
            safe_ids = local_ids.clamp_min(0)

        scores = self._score_gmm_neighbors(
            target_chunk,
            scene_source_coords[safe_ids],
            scene_opacity[safe_ids],
            scene_scale[safe_ids],
            scene_rotation[safe_ids],
            valid_mask)
        return scores, valid_mask.any(dim=1)


    def _evaluate_gmm_field_local_window(self, target_x, gmm_params):
        keep_prob = target_x.features.new_zeros((len(target_x.features),))
        opacity = gmm_params['opacity']
        scale = gmm_params['scale']
        rotation = gmm_params['rotation']
        source_coords = gmm_params['coords']
        spacing = gmm_params['spacing']
        chunk_size = max(int(self.gmm_chunk_size), 1)
        primary_radius = max(int(self.gmm_local_window_radius), 0)
        fallback_mode = self.gmm_local_fallback
        fallback_radius = max(primary_radius, int(self.gmm_local_fallback_radius))
        primary_offsets = self._gmm_local_offsets(primary_radius, target_x.features.device)
        fallback_offsets = None
        if fallback_mode in ('expand', 'nearest_missing') and fallback_radius > primary_radius:
            fallback_offsets = self._gmm_local_offsets(fallback_radius, target_x.features.device)
        elif fallback_mode not in ('none', None, 'expand', 'nearest_missing'):
            raise ValueError(f'Unsupported GMM local fallback: {fallback_mode}')
        cell_scale = max(float(self.gmm_local_cell_size_scale), 1e-6)
        cell_size = (spacing * cell_scale).clamp_min(1e-6)

        for source_perm, target_perm in zip(gmm_params['permutations'], target_x.decomposition_permutations):
            if len(source_perm) == 0 or len(target_perm) == 0:
                continue

            scene_source_coords = source_coords[source_perm]
            scene_opacity = opacity[source_perm]
            scene_scale = scale[source_perm]
            scene_rotation = rotation[source_perm]
            scene_target_coords = target_x.coordinates[target_perm][:, 1:].float() * self.voxel_size
            source_cells = torch.floor(scene_source_coords / cell_size).long()
            target_cells = torch.floor(scene_target_coords / cell_size).long()
            lookup_radius = fallback_radius if fallback_offsets is not None else primary_radius
            min_cell = torch.min(source_cells.min(dim=0).values,
                                 target_cells.min(dim=0).values) - lookup_radius
            max_cell = torch.max(source_cells.max(dim=0).values,
                                 target_cells.max(dim=0).values) + lookup_radius
            grid_shape = max_cell - min_cell + 1
            lookup = self._build_gmm_cell_lookup(source_cells, min_cell, grid_shape)

            for start in range(0, len(scene_target_coords), chunk_size):
                end = min(start + chunk_size, len(scene_target_coords))
                target_chunk = scene_target_coords[start:end]
                chunk_target_cells = target_cells[start:end]
                chunk_scores, found_mask = self._score_gmm_local_candidates(
                    target_chunk, chunk_target_cells, scene_source_coords,
                    scene_opacity, scene_scale, scene_rotation, primary_offsets,
                    min_cell, grid_shape, lookup)

                if fallback_mode == 'expand' and fallback_offsets is not None:
                    chunk_scores, found_mask = self._score_gmm_local_candidates(
                        target_chunk, chunk_target_cells, scene_source_coords,
                        scene_opacity, scene_scale, scene_rotation, fallback_offsets,
                        min_cell, grid_shape, lookup)
                elif fallback_mode == 'nearest_missing':
                    missing_mask = ~found_mask
                    if missing_mask.any() and fallback_offsets is not None:
                        fallback_scores, fallback_found = self._score_gmm_local_candidates(
                            target_chunk[missing_mask], chunk_target_cells[missing_mask],
                            scene_source_coords, scene_opacity, scene_scale,
                            scene_rotation, fallback_offsets, min_cell, grid_shape,
                            lookup)
                        chunk_scores = chunk_scores.clone()
                        found_mask = found_mask.clone()
                        chunk_scores[missing_mask] = fallback_scores
                        found_mask[missing_mask] = fallback_found
                        missing_mask = ~found_mask
                    if missing_mask.any():
                        k = min(max(int(self.gmm_knn_k), 1), len(scene_source_coords))
                        nn_ids = torch.cdist(
                            target_chunk[missing_mask], scene_source_coords).topk(
                                k, largest=False, sorted=False).indices
                        fallback_scores = self._score_gmm_neighbors(
                            target_chunk[missing_mask],
                            scene_source_coords[nn_ids],
                            scene_opacity[nn_ids],
                            scene_scale[nn_ids],
                            scene_rotation[nn_ids])
                        if chunk_scores.data_ptr() == keep_prob[target_perm[start:end]].data_ptr():
                            chunk_scores = chunk_scores.clone()
                        chunk_scores[missing_mask] = fallback_scores

                keep_prob[target_perm[start:end]] = chunk_scores
        return keep_prob


    def _evaluate_gmm_field(self, target_x, gmm_params):
        if self.gmm_neighbor_backend == 'cdist':
            keep_prob = self._evaluate_gmm_field_cdist(target_x, gmm_params)
        elif self.gmm_neighbor_backend in ('local_window', 'voxel_window'):
            keep_prob = self._evaluate_gmm_field_local_window(target_x, gmm_params)
        else:
            raise ValueError(f'Unsupported GMM neighbor backend: {self.gmm_neighbor_backend}')

        volume, opacity_entropy = self._gmm_regularizers(gmm_params)
        return keep_prob.clamp(0, 1), volume, opacity_entropy


    def _apply_gmm_training_gate(self, x, keep_prob):
        if self.current_epoch < self.gmm_warmup_epochs:
            gate = torch.ones_like(keep_prob)
        else:
            gate = self.gmm_train_gate_floor + (1 - self.gmm_train_gate_floor) * keep_prob
        return self._sparse_like(x, x.features * gate.unsqueeze(1))


    def _make_gmm_prune_mask(self, x, keep_prob):
        prune_mask = keep_prob.new_zeros((len(keep_prob),), dtype=torch.bool)
        for permutation in x.decomposition_permutations:
            if len(permutation) == 0:
                continue
            score = keep_prob[permutation]
            mask = score > self.gmm_keep_threshold
            min_keep = min(int(self.gmm_min_keep), len(score))
            max_keep = min(int(self.gmm_max_keep), len(score))
            if mask.sum() < min_keep:
                ids = torch.topk(score, min_keep, sorted=False).indices
                mask = torch.zeros_like(mask)
                mask[ids] = True
            if max_keep > 0 and mask.sum() > max_keep:
                kept_scores = score.masked_fill(~mask, -1)
                ids = torch.topk(kept_scores, max_keep, sorted=False).indices
                mask = torch.zeros_like(mask)
                mask[ids] = True
            if mask.sum() == 0:
                mask[torch.argmax(score)] = True
            prune_mask[permutation[mask]] = True
        return prune_mask


    def _prune_gmm_inference(self, x, keep_prob):
        with torch.no_grad():
            prune_mask = self._make_gmm_prune_mask(x, keep_prob)
        if prune_mask.sum() == 0:
            return None
        return self.pruning(x, prune_mask)


    def _make_gaussian_prune_mask(self, keep_prob, x):
        return self._make_gmm_prune_mask(x, keep_prob.reshape(-1))


    def _prune_by_gaussian(self, x, keep_prob):
        return self._prune_gmm_inference(x, keep_prob.reshape(-1))


    def forward(self, x, gt_bboxes, gt_labels, img_metas):

        bboxes_level = []
        bboxes_state = []
        if self.assign_type == 'volume':
            for idx in range(len(img_metas)):
                bbox = gt_bboxes[idx]
                bbox_state = torch.cat((bbox.gravity_center, bbox.tensor[:, 3:]), dim=1)
                bbox_level = torch.zeros([len(bbox), 1])
                downsample_times = [5,4,3]
                for n in range(len(bbox)):
                    bbox_volume = bbox_state[n][3] * bbox_state[n][4] * bbox_state[n][5]
                    for i in range(len(downsample_times)):
                        if bbox_volume > self.volume_threshold * (self.voxel_size * 2 ** downsample_times[i]) ** 3:
                            bbox_level[n] = 3 - i
                            break
                bboxes_level.append(bbox_level)
                bbox_state = torch.cat((bbox_level, bbox_state), dim=1)
                bboxes_state.append(bbox_state)
        elif self.assign_type == 'label':
            for idx in range(len(img_metas)):
                bbox = gt_bboxes[idx]
                bbox_label = gt_labels[idx]
                label2level = gt_labels[idx].new_tensor(self.label2level)
                bbox_state = torch.cat((bbox.gravity_center, bbox.tensor[:, 3:]), dim=1)
                bbox_level = label2level[bbox_label].to(bbox_state.device).unsqueeze(1)
                bboxes_level.append(bbox_level)
                bbox_state = torch.cat((bbox_level, bbox_state), dim=1)
                bboxes_state.append(bbox_state)
        bbox_preds, cls_preds, points = [], [], []
        keep_gts = []
        keep_preds, prune_masks = [], []
        gmm_volume_losses, gmm_sparsity_losses = [], []
        prune_mask = None
        inputs = x
        x = inputs[-1]
        gmm_params = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                if self.gaussian_pruning_enabled:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                    keep_prob, volume_loss, sparsity_loss = self._evaluate_gmm_field(x, gmm_params)
                    prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)
                    keep_gt, keeps = [], []
                    keep_logit = torch.logit(keep_prob.clamp(1e-4, 1 - 1e-4)).unsqueeze(1)
                    for permutation in x.decomposition_permutations:
                        keep_gt.append(prune_mask[permutation])
                        keeps.append(keep_logit[permutation])
                    keep_gts.append(keep_gt)
                    keep_preds.append(keeps)
                    gmm_volume_losses.append(volume_loss)
                    gmm_sparsity_losses.append(sparsity_loss)
                    x = self._apply_gmm_training_gate(x, keep_prob)
                else:
                    prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)
                    keep_gt = []
                    for permutation in out.decomposition_permutations:
                        keep_gt.append(prune_mask[permutation])
                    keep_gts.append(keep_gt)
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                    x = self._prune_training(x, prune_training_keep)

            if i > 0:
                if self.gaussian_pruning_enabled:
                    gmm_params = self._decode_gmm_params(x, i - 1)
                else:
                    keep_scores = self.keep_conv[i-1](x)
                    prune_training_keep = ME.SparseTensor(
                                        -keep_scores.features,
                                        coordinate_map_key=keep_scores.coordinate_map_key,
                                        coordinate_manager=keep_scores.coordinate_manager)
                    keep_pred = keep_scores.features
                    prune_inference = keep_pred
                    keeps = []
                    for permutation in x.decomposition_permutations:
                        keeps.append(keep_pred[permutation])
                    keep_preds.append(keeps)
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            bbox_pred, cls_pred, point, prune_training = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

        return (bbox_preds[::-1], cls_preds[::-1], points[::-1], keep_preds[::-1],
                keep_gts[::-1], bboxes_level, gmm_volume_losses, gmm_sparsity_losses)


    def _prune_inference(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > self.prune_threshold
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x


    def _prune_training(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x


    @torch.no_grad()
    def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (cur_level - 1):
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for idx, permutation in enumerate(input.decomposition_permutations):
            point = input.coordinates[permutation][:, 1:] * self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:] * self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                            point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = self.r
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask


    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)


    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     bboxes_level,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, bboxes_level, img_meta)

        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0


        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds.sum().reshape(1)
        return bbox_loss, cls_loss, pos_mask


    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
              gmm_volume_losses=None, gmm_sparsity_losses=None):
        bbox_losses, cls_losses, pos_masks = [], [], []
        zero_loss = self.bbox_conv.kernel.sum() * 0

        keep_losses = zero_loss
        if len(keep_preds) > 0:
            for i in range(len(img_metas)):
                k_loss = zero_loss
                keep_pred = [x[i] for x in keep_preds]
                keep_gt = [x[i] for x in keep_gts]
                for j in range(len(keep_preds)):
                    pred = keep_pred[j]
                    gt = keep_gt[j].long()
                    if gt.numel() == 0:
                        continue
                    avg_factor = gt.sum() if gt.sum() != 0 else gt.new_tensor(gt.numel())
                    keep_loss = self.keep_loss(pred, gt, avg_factor=avg_factor)
                    k_loss = torch.mean(keep_loss) / len(keep_preds) + k_loss
                keep_losses = keep_losses + k_loss

        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                bboxes_level=bboxes_level[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)

        bbox_loss = torch.mean(torch.cat(bbox_losses)) if len(bbox_losses) > 0 else zero_loss
        cls_loss = torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)).clamp(min=1)
        loss_dict = dict(
            bbox_loss=bbox_loss,
            cls_loss=cls_loss,
            keep_loss=self.gmm_loss_weight * keep_losses / len(img_metas))

        if self.gaussian_pruning_enabled:
            if gmm_volume_losses:
                loss_dict['loss_gmm_volume'] = self.gmm_volume_loss_weight * torch.stack(gmm_volume_losses).mean()
            else:
                loss_dict['loss_gmm_volume'] = zero_loss
            if gmm_sparsity_losses:
                loss_dict['loss_gmm_sparsity'] = self.gmm_sparsity_loss_weight * torch.stack(gmm_sparsity_losses).mean()
            else:
                loss_dict['loss_gmm_sparsity'] = zero_loss
        return loss_dict


    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        (bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level,
         gmm_volume_losses, gmm_sparsity_losses) = self(x, gt_bboxes, gt_labels, img_metas)
        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
                          gmm_volume_losses, gmm_sparsity_losses)


    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels


    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def forward_test(self, x, img_metas):
        inputs = x
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        keep_scores = None
        gmm_params = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                if self.gaussian_pruning_enabled:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                    keep_prob, _, _ = self._evaluate_gmm_field(x, gmm_params)
                    x = self._prune_gmm_inference(x, keep_prob)
                    if x is None:
                        break
                else:
                    x = self._prune_inference(x, prune_inference)
                    if x != None:
                        x = self.__getattr__(f'up_block_{i + 1}')(x)
                        coords = x.coordinates.float()
                        x_level_features = inputs[i].features_at_coordinates(coords)
                        x_level = ME.SparseTensor(features=x_level_features,
                                                  coordinate_map_key=x.coordinate_map_key,
                                                  coordinate_manager=x.coordinate_manager)
                        x = x + x_level
                    else:
                        break

            if i > 0:
                if self.gaussian_pruning_enabled:
                    gmm_params = self._decode_gmm_params(x, i - 1)
                else:
                    keep_scores = self.keep_conv[i-1](x)
                    keep_pred = keep_scores.features
                    prune_inference = keep_pred

            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            bbox_pred, cls_pred, point, prune_training = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

        return self._get_bboxes(bbox_preds[::-1], cls_preds[::-1], points[::-1], img_metas)


@BBOX_ASSIGNERS.register_module()
class DSPAssigner:
    def __init__(self, top_pts_threshold):
        # top_pts_threshold: per box
        self.top_pts_threshold = top_pts_threshold

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, bboxes_level, img_meta):
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(points.device)
        bboxes_level = bboxes_level.squeeze(1).to(points.device)

        nearest_distances = points.new_full((n_points,), 1e8)
        nearest_ids = torch.full((n_points,), -1, dtype=torch.long, device=points.device)
        candidate_distances = points.new_full((n_points,), 1e8)
        candidate_ids = torch.full((n_points,), -1, dtype=torch.long, device=points.device)

        L0 = 0.01 * 2 ** 2
        p = 7
        topk = min(self.top_pts_threshold + 1, n_points)
        half_sizes = p * (L0 * 2 ** bboxes_level) / 2
        box_chunk_size = 16

        for start in range(0, n_boxes, box_chunk_size):
            end = min(start + box_chunk_size, n_boxes)
            centers = boxes[start:end, :3]
            offsets = points.unsqueeze(1) - centers.unsqueeze(0)
            distances = torch.sum(torch.pow(offsets, 2), dim=-1)

            nearest_values, nearest_local_ids = distances.min(dim=1)
            update_nearest = nearest_values < nearest_distances
            nearest_distances[update_nearest] = nearest_values[update_nearest]
            nearest_ids[update_nearest] = nearest_local_ids[update_nearest] + start

            valid_mask = levels.unsqueeze(1) == bboxes_level[start:end].unsqueeze(0)
            valid_mask = valid_mask & (offsets.abs().max(dim=2).values < half_sizes[start:end].unsqueeze(0))
            filtered_distances = torch.where(valid_mask, distances, distances.new_full((), 1e8))
            topk_distances = torch.topk(filtered_distances, topk, largest=False, dim=0).values[-1]
            topk_mask = filtered_distances < topk_distances.unsqueeze(0)
            filtered_distances = torch.where(topk_mask, distances, distances.new_full((), 1e8))

            candidate_values, candidate_local_ids = filtered_distances.min(dim=1)
            update_candidate = candidate_values < candidate_distances
            candidate_distances[update_candidate] = candidate_values[update_candidate]
            candidate_ids[update_candidate] = candidate_local_ids[update_candidate] + start

        assigned_ids = torch.full((n_points,), -1, dtype=torch.long, device=points.device)
        assigned_mask = candidate_ids == nearest_ids
        assigned_ids[assigned_mask] = candidate_ids[assigned_mask]
        return assigned_ids
