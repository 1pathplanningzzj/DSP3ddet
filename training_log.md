# Training Log

## 2026-06-08

- 16:14
  更新 [multilevel_head.py](/home/czy22/zzj/DSPDet3D/dspdet3d/models/dense_heads/multilevel_head.py)。
  新增 GMM 推理分数融合接口，支持 `score_fusion_mode='mul'/'logit'`，并在 `nms_pre` 前把 `keep_prob` 接入最终排序。

- 16:14
  更新 [multilevel_head.py](/home/czy22/zzj/DSPDet3D/dspdet3d/models/dense_heads/multilevel_head.py)。
  去掉旧的 `soft_then_hard` 中途硬切换语义：老配置如果写了 `train_prune_mode='soft_then_hard'`，现在按纯 `soft` 处理，避免训练中后期直接切到 hard prune。

- 22:00
  新增配置 [dspdet3d_scannet-3d-22class_gmm_soft.py](/home/czy22/zzj/DSPDet3D/configs/dspdet3d/dspdet3d_scannet-3d-22class_gmm_soft.py)。
  本次实验固定为：
  `gaussian_pruning.train_prune_mode='soft'`
  `gaussian_pruning.test_prune_mode='soft'`
  单卡 `GPU 0`，`work_dir=work_dirs/scannet_md40_3dgmm_soft_train_soft_test_gpu0`

- 22:19
  启动新的 ScanNet 训练：
  配置文件 [dspdet3d_scannet-3d-22class_gmm_soft.py](/home/czy22/zzj/DSPDet3D/configs/dspdet3d/dspdet3d_scannet-3d-22class_gmm_soft.py)
  日志目录 [scannet_md40_3dgmm_soft_train_soft_test_gpu0](/home/czy22/zzj/DSPDet3D/work_dirs/scannet_md40_3dgmm_soft_train_soft_test_gpu0)
  `tmux` 会话名：`scannet_gmm_soft_gpu0`
  训练日志文件：[20260608_221930.log](/home/czy22/zzj/DSPDet3D/work_dirs/scannet_md40_3dgmm_soft_train_soft_test_gpu0/20260608_221930.log)

- 22:20
  更新 [command.md](/home/czy22/zzj/DSPDet3D/command.md)。
  追加这次纯 soft 训练的 `tmux` 启动命令、环境变量和日志查看方式，方便后续复现。

## 2026-06-10

- 19:20
  更新 [multilevel_head.py](/home/czy22/zzj/DSPDet3D/dspdet3d/models/dense_heads/multilevel_head.py)。
  修复并补强 Gaussian pruning 主流程：
  修正测试阶段 `keep_prob` 与输出体素坐标的对齐；
  修正非高斯分支的错层 `decomposition_permutations` 引用；
  测试软门控绕开 warmup；
  增加 `target_keep_ratios`、可学习中心偏移 `offset_conv`、`noisy_or/topk_mean` 聚合、预算损失回退和 `loss_gmm_offset`。

- 19:20
  更新配置 [dspdet3d_scannet-3d-22class_gmm_soft.py](/home/czy22/zzj/DSPDet3D/configs/dspdet3d/dspdet3d_scannet-3d-22class_gmm_soft.py)。
  当前实验在原纯 soft 版本基础上新增：
  `budget_loss_weight=0.01`
  `target_keep_ratios=[0.2, 0.35, 0.5]`，按粗到细解释
  `offset_range=1.0`
  `offset_loss_weight=0.001`
  `aggregation='noisy_or'`
  训练和测试仍保持：
  `gaussian_pruning.train_prune_mode='soft'`
  `gaussian_pruning.test_prune_mode='soft'`

- 19:20
  启动新的 ScanNet 训练：
  配置文件 [dspdet3d_scannet-3d-22class_gmm_soft.py](/home/czy22/zzj/DSPDet3D/configs/dspdet3d/dspdet3d_scannet-3d-22class_gmm_soft.py)
  日志目录 [scannet_md40_3dgmm_soft_budget_offset_gpu1](/home/czy22/zzj/DSPDet3D/work_dirs/scannet_md40_3dgmm_soft_budget_offset_gpu1)
  `tmux` 会话名：`scannet_gmm_soft_gpu1_budget`
  训练日志文件：[20260610_192035.log](/home/czy22/zzj/DSPDet3D/work_dirs/scannet_md40_3dgmm_soft_budget_offset_gpu1/20260610_192035.log)
  单卡 `GPU 1`，用于验证预算约束、可学习偏移和多高斯聚合在 soft gate 模式下的训练稳定性。


