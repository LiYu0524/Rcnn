#!/usr/bin/env python3
"""
MMDetection环境设置脚本
用于在VOC数据集上训练Mask R-CNN和Sparse R-CNN
"""

import os
import subprocess
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path

def run_command(cmd, check=True):
    """运行shell命令"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        sys.exit(1)
    return result

def install_mmdetection():
    """安装MMDetection框架"""
    print("开始安装MMDetection...")
    
    # 安装依赖
    dependencies = [
        "torch>=1.8.0",
        "torchvision>=0.9.0", 
        "mmcv-full>=1.4.0",
        "mmdet>=2.25.0",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pillow",
        "numpy",
        "scipy"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}")
    
    # 克隆mmdetection仓库
    if not os.path.exists("mmdetection_repo"):
        run_command("git clone https://github.macc.cc/open-mmlab/mmdetection.git mmdetection_repo")
    
    # 安装mmdetection
    os.chdir("mmdetection_repo")
    run_command("pip install -v -e .")
    os.chdir("..")
    
    print("MMDetection安装完成!")

def check_voc_dataset():
    """检查VOC数据集是否存在"""
    print("检查VOC数据集...")
    
    data_dir = Path("data/VOCdevkit")
    voc2007_dir = data_dir / "VOC2007"
    voc2012_dir = data_dir / "VOC2012"
    
    # 检查必要的目录和文件是否存在
    required_paths = [
        voc2007_dir / "JPEGImages",
        voc2007_dir / "Annotations", 
        voc2007_dir / "ImageSets" / "Main",
        voc2012_dir / "JPEGImages",
        voc2012_dir / "Annotations",
        voc2012_dir / "ImageSets" / "Main"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        print(f"❌ VOC数据集不完整，缺少以下路径:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    else:
        print("✅ VOC数据集检查完成，数据集完整!")
        return True

def download_voc_dataset():
    """下载VOC数据集（仅在数据集不存在时）"""
    # 首先检查数据集是否已存在
    if check_voc_dataset():
        print("VOC数据集已存在，跳过下载步骤。")
        return
    
    print("开始下载VOC数据集...")
    
    data_dir = Path("data/VOCdevkit")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # VOC2007和VOC2012下载链接
    voc_urls = {
        "VOC2007_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "VOC2007_test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "VOC2012_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    }
    
    for name, url in voc_urls.items():
        tar_path = data_dir / f"{name}.tar"
        if not tar_path.exists():
            print(f"下载 {name}...")
            try:
                urllib.request.urlretrieve(url, tar_path)
                
                print(f"解压 {name}...")
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(data_dir)
                    
                # 删除tar文件以节省空间
                tar_path.unlink()
            except Exception as e:
                print(f"下载或解压 {name} 时出错: {e}")
                if tar_path.exists():
                    tar_path.unlink()
                continue
    
    # 再次检查数据集完整性
    if check_voc_dataset():
        print("VOC数据集下载并验证完成!")
    else:
        print("❌ VOC数据集下载后验证失败，请检查网络连接或手动下载数据集。")

def setup_configs():
    """设置训练配置文件"""
    print("创建配置文件...")
    
    # 创建配置目录
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Mask R-CNN配置
    mask_rcnn_config = """
# Mask R-CNN配置文件
_base_ = [
    '../mmdetection_repo/configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../mmdetection_repo/configs/_base_/datasets/coco_instance.py',
    '../mmdetection_repo/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection_repo/configs/_base_/default_runtime.py'
]

# 模型配置
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),  # VOC有20个类别
        mask_head=dict(num_classes=20)
    )
)

# 数据集配置
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

# 优化器配置
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# 运行时配置
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metric=['bbox', 'segm'])
"""
    
    with open(config_dir / "mask_rcnn_voc.py", "w") as f:
        f.write(mask_rcnn_config)
    
    print("配置文件创建完成!")

def main():
    """主函数"""
    print("开始设置MMDetection环境...")
    
    # 创建项目目录结构
    dirs = ["data", "configs", "work_dirs", "results", "visualizations"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # 安装MMDetection
    install_mmdetection()
    
    # 下载VOC数据集
    download_voc_dataset()
    
    # 设置配置文件
    setup_configs()
    
    print("环境设置完成!")
    print("接下来可以开始训练模型了。")

if __name__ == "__main__":
    main() 