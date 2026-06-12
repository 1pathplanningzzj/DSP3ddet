voxel_size = 0.01
n_points = 100000

model = dict(
    type='DSPDet3D',
    voxel_size=voxel_size,
    backbone=dict(
        type='DSPBackbone',
        in_channels=3,
        max_channels=128,
        depth=34,
        pool=False,
        norm='batch'),
    head=dict(
        type='DSPHead',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
        n_reg_outs=6,
        n_classes=22,
        voxel_size=voxel_size,
        pts_prune_threshold=100000,
        assigner=dict(type='DSPAssigner', top_pts_threshold=6),
        assign_type='volume',
        volume_threshold=27,
        r=7,
        prune_threshold=0.3,
        gaussian_pruning=dict(
            enabled=True,
            num_primitives=1,
            projection='nearest',
            keep_threshold=0.5,
            min_keep=1000,
            max_keep=100000,
            warmup_epochs=1,
            loss_weight=0.01,
            gmm_loss_weight=0.01,
            primitive_loss_weight=0.0,
            sigma_scale=0.5,
            sigma_min=0.1,
            sigma_max=2.0,
            scale_min=0.1,
            scale_max=2.0,
            mean_offset_scale=1.5,
            target_edge_prob=0.5,
            chunk_size=2048,
            knn_k=3,
            neighbor_backend='local_window',
            local_window_radius=1,
            local_cell_size_scale=1.0,
            local_dense_max_cells=2000000,
            local_fallback='nearest_missing',
            local_fallback_radius=2,
            train_gate_floor=0.2,
            test_prune_mode='soft',
            train_prune_mode='soft',
            target_type='soft_support',
            soft_target_scale=1.0,
            soft_target_norm_mode='box_adaptive',
            soft_target_box_scale=0.5,
            soft_target_min_scale=0.5,
            soft_pos_weight=4.0,
            budget_loss_weight=0.01,
            target_keep_ratios=[0.2, 0.35, 0.5],
            offset_range=1.0,
            offset_loss_weight=0.001,
            aggregation='noisy_or',
            volume_loss_weight=0.002,
            opacity_sparsity_loss_weight=0.005,
            score_fusion_mode='none',
            score_fusion_weight=0.2),
        bbox_loss=dict(
            type='AxisAlignedIoULoss2', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    dict(type='GaussianPruningEpochHook')
]
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metric='mAP')
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/scannet_md40_3dgmm_soft_train_soft_test_gpu0'
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'ScanNetDataset'
data_root = 'data/ScanNet-md40/mmdet_scannet/'
class_names = ('bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair',
               'cup', 'curtain', 'desk', 'door', 'dresser', 'keyboard', 'lamp',
               'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool',
               'table', 'toilet', 'wardrobe')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSample', num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.02, 0.02],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair',
                     'cup', 'curtain', 'desk', 'door', 'dresser', 'keyboard',
                     'lamp', 'laptop', 'monitor', 'night_stand', 'plant',
                     'sofa', 'stool', 'table', 'toilet', 'wardrobe')),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=('bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
                             'chair', 'cup', 'curtain', 'desk', 'door',
                             'dresser', 'keyboard', 'lamp', 'laptop',
                             'monitor', 'night_stand', 'plant', 'sofa',
                             'stool', 'table', 'toilet', 'wardrobe'),
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='ScanNetDataset',
            data_root='data/ScanNet-md40/mmdet_scannet/',
            ann_file='data/ScanNet-md40/mmdet_scannet/scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type='ScanNetDataset',
        data_root='data/ScanNet-md40/mmdet_scannet/',
        ann_file='data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type='ScanNetDataset',
        data_root='data/ScanNet-md40/mmdet_scannet/',
        ann_file='data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
