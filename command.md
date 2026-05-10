# DSPDet3D 复现命令

## 1. 进入环境

Anaconda 的 `activate` 脚本里有旧路径，所以用 conda hook 激活：

```bash
cd /home/czy22/zzj/DSPDet3D

eval "$(/data/czy22/anaconda3/bin/python /data/czy22/anaconda3/bin/conda shell.bash hook)"
conda activate /data/czy22/anaconda3/envs/dspdet3d

export LD_LIBRARY_PATH=/data/czy22/anaconda3/envs/dspdet3d/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/czy22/zzj/DSPDet3D:$PYTHONPATH
export OMP_NUM_THREADS=12
```

验证环境：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available())
import MinkowskiEngine as ME
print('ME:', ME.__version__)
import mmcv, mmdet, mmseg, mmdet3d
print('mmcv:', mmcv.__version__)
print('mmdet:', mmdet.__version__)
print('mmseg:', mmseg.__version__)
print('mmdet3d:', mmdet3d.__version__)
import dspdet3d
print('DSPDet3D import ok')
PY
```

当前已验证版本：

```text
torch 1.13.1+cu117
MinkowskiEngine 0.5.4
mmcv-full 1.7.0
mmdet 2.28.2
mmsegmentation 0.30.0
mmdet3d 1.0.0rc6
```

如果后续 mmcv 兼容报错，可降级：

```bash
pip uninstall -y mmcv-full mmcv
mim install "mmcv-full==1.6.2" \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

## 2. 数据和权重目录

数据放在 `/data`，项目里只放软链接：

```bash
mkdir -p /data/czy22/DSPDet3D_data/downloads
mkdir -p /data/czy22/DSPDet3D_data/ScanNet-md40
mkdir -p /data/czy22/DSPDet3D_data/checkpoints/scannet_md40

cd /home/czy22/zzj/DSPDet3D
mv data/ScanNet-md40 data/ScanNet-md40_scripts_backup 2>/dev/null || true
ln -sfn /data/czy22/DSPDet3D_data/ScanNet-md40 data/ScanNet-md40
mkdir -p work_dirs
ln -sfn /data/czy22/DSPDet3D_data/checkpoints/scannet_md40 work_dirs/scannet_md40
```

## 3. 后台下载 ScanNet processed data

清华云目录 token：`2786204cfff94b408ea6`。

ScanNet-md40 processed data 文件：

```text
mmdet_scannet.tar.00
mmdet_scannet.tar.01
mmdet_scannet.tar.02
```

后台下载：

```bash
mkdir -p /data/czy22/DSPDet3D_data/downloads
cd /data/czy22/DSPDet3D_data/downloads

nohup bash -c '
set -e
wget -c -O mmdet_scannet.tar.00 "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fmmdet_scannet.tar.00&dl=1"
wget -c -O mmdet_scannet.tar.01 "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fmmdet_scannet.tar.01&dl=1"
wget -c -O mmdet_scannet.tar.02 "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fmmdet_scannet.tar.02&dl=1"
' > scannet_download.log 2>&1 &
```

查看进度：

```bash
tail -f /data/czy22/DSPDet3D_data/downloads/scannet_download.log
ls -lh /data/czy22/DSPDet3D_data/downloads/mmdet_scannet.tar.*
```

下载完成后合并解压：

```bash
cd /data/czy22/DSPDet3D_data/downloads
cat mmdet_scannet.tar.00 mmdet_scannet.tar.01 mmdet_scannet.tar.02 > mmdet_scannet.tar

tar -xf mmdet_scannet.tar -C /data/czy22/DSPDet3D_data/ScanNet-md40

find /data/czy22/DSPDet3D_data/ScanNet-md40 -maxdepth 3 \
  \( -name 'scannet_infos_train.pkl' -o -name 'scannet_infos_val.pkl' \) -print
```

最终期望目录里有：

```text
points/
instance_mask/
semantic_mask/
seg_info/
scannet_infos_train.pkl
scannet_infos_val.pkl
```

## 4. 下载 checkpoint

ScanNet-md40 checkpoint：

```bash
mkdir -p /data/czy22/DSPDet3D_data/checkpoints/scannet_md40
wget -c -O /data/czy22/DSPDet3D_data/checkpoints/scannet_md40/latest.pth \
  "https://cloud.tsinghua.edu.cn/f/bd49db94cb7548beba63/?dl=1"
```

## 5. 下载并运行 demo

下载 demo 权重和点云：

```bash
cd /home/czy22/zzj/DSPDet3D

wget -c -O demo/scannet.ply \
  "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fscannet.ply&dl=1"

wget -c -O demo/dspdet3d_demo.pth \
  "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fdspdet3d_demo.pth&dl=1"

mkdir -p demo/demo_data
bash demo/demo.sh demo/scannet.ply demo/config_room.py
```

结果目录：

```text
demo/demo_results/scannet/
```

## 6. 评测 ScanNet-md40

解压后的真实数据目录是：

```text
data/ScanNet-md40/mmdet_scannet/
```

配置文件里的 `ann_file` 已经在加载时拼成了 `/path/to/.pkl/scannet_infos_val.pkl`，所以只覆盖顶层 `data_root` 不够，要覆盖 `data.test.data_root` 和 `data.test.ann_file`。

单卡评测：

```bash
cd /home/czy22/zzj/DSPDet3D

export LD_LIBRARY_PATH=/data/czy22/anaconda3/envs/dspdet3d/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/czy22/zzj/DSPDet3D:$PYTHONPATH
export OMP_NUM_THREADS=12

CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class.py \
  work_dirs/scannet_md40/latest.pth \
  1 \
  --eval mAP \
  --cfg-options \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

如果使用 2 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_test.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class.py \
  work_dirs/scannet_md40/latest.pth \
  2 \
  --eval mAP \
  --cfg-options \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

## 7. 改进模型、训练自己的 checkpoint 并测评

建议流程：先复现 baseline，再复制一份配置做改动，训练后用同一个 val 集测 mAP，最后和论文 checkpoint 对比。

### 7.1 复制自己的配置

不要直接改原始配置，先复制一份：

```bash
cd /home/czy22/zzj/DSPDet3D

cp configs/dspdet3d/dspdet3d_scannet-3d-22class.py \
   configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py
```

之后主要改：

```text
configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py
```

可以优先尝试这些位置：

```python
voxel_size = .01
n_points = 100000

model = dict(
    backbone=dict(
        depth=34,
        max_channels=128,
    ),
    head=dict(
        prune_threshold=0.3,
        pts_prune_threshold=100000,
        r=7,
        volume_threshold=27,
    )
)

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
```

常见实验方向：

- `prune_threshold`：速度/精度权衡，建议先试 `0.2 / 0.3 / 0.4 / 0.5`
- `voxel_size`：更小可能更精细，但更慢、更吃显存
- `n_points`：更多点可能提升精度，但训练更慢
- `max_channels` / `depth`：增强 backbone
- `max_epochs` / `optimizer.lr`：训练更久或调学习率
- 数据增强：`RandomFlip3D`、`GlobalRotScaleTrans`

### 7.2 从头训练自己的模型

单卡训练：

```bash
cd /home/czy22/zzj/DSPDet3D

export LD_LIBRARY_PATH=/data/czy22/anaconda3/envs/dspdet3d/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/czy22/zzj/DSPDet3D:$PYTHONPATH
export OMP_NUM_THREADS=12

CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py \
  1 \
  --work-dir work_dirs/scannet_md40_my \
  --cfg-options \
  data.train.dataset.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.train.dataset.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_train.pkl \
  data.val.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.val.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

2 卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py \
  2 \
  --work-dir work_dirs/scannet_md40_my \
  --cfg-options \
  data.train.dataset.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.train.dataset.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_train.pkl \
  data.val.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.val.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

输出目录：

```text
work_dirs/scannet_md40_my/
```

里面会生成：

```text
epoch_1.pth
epoch_2.pth
...
latest.pth
```

### 7.3 测评自己训练的模型

测 `latest.pth`：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py \
  work_dirs/scannet_md40_my/latest.pth \
  1 \
  --eval mAP \
  --cfg-options \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

测指定 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py \
  work_dirs/scannet_md40_my/epoch_12.pth \
  1 \
  --eval mAP \
  --cfg-options \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

重点看输出里的：

```text
mAP@0.25
mAP@0.50
```

### 7.4 基于论文权重微调

如果想从官方 checkpoint 开始，而不是从头训：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh \
  configs/dspdet3d/dspdet3d_scannet-3d-22class_my.py \
  1 \
  --work-dir work_dirs/scannet_md40_my_finetune \
  --cfg-options \
  load_from=work_dirs/scannet_md40/latest.pth \
  optimizer.lr=0.0001 \
  data.train.dataset.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.train.dataset.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_train.pkl \
  data.val.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.val.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl \
  data.test.data_root=data/ScanNet-md40/mmdet_scannet/ \
  data.test.ann_file=data/ScanNet-md40/mmdet_scannet/scannet_infos_val.pkl
```

### 7.5 建议的第一组实验

先不要大改结构，做最小可控实验：

1. 跑官方 checkpoint，记录 baseline mAP 和推理速度。
2. 复制 config，改 `prune_threshold=0.2`，用官方 checkpoint 直接测。
3. 改 `prune_threshold=0.4`，直接测。
4. 改 `prune_threshold=0.5`，直接测。
5. 选择速度/精度更合适的阈值后，再考虑微调训练。

只改 `prune_threshold` 通常不需要重新训练，可以直接用同一个 checkpoint 测试不同速度/精度权衡。

## 8. 常见问题

### `libgfortran.so.3` 找不到

先设置：

```bash
export LD_LIBRARY_PATH=/data/czy22/anaconda3/envs/dspdet3d/lib:$LD_LIBRARY_PATH
```

如果还不行，安装旧 runtime 或重编译 MinkowskiEngine。

### `demo/scannet.ply` 不存在

先下载 demo 点云：

```bash
wget -c -O demo/scannet.ply \
  "https://cloud.tsinghua.edu.cn/d/2786204cfff94b408ea6/files/?p=%2Fscannet.ply&dl=1"
```

### `MMCV==1.7.2 is incompatible`

`mmdet3d==1.0.0rc6` 不兼容 `mmcv-full==1.7.2`。使用 `mmcv-full==1.7.0` 或 `1.6.2`。
