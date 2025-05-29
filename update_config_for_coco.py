#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新MMDetection配置文件以使用转换后的COCO格式数据
"""

import os
import json
from pathlib import Path


def update_mask_rcnn_config():
    """更新Mask R-CNN配置文件"""
    
    # 检查COCO数据是否存在
    coco_data_root = "./data/coco_format"
    dataset_info_file = os.path.join(coco_data_root, "dataset_info.json")
    
    if not os.path.exists(dataset_info_file):
        print(f"❌ 数据集信息文件不存在: {dataset_info_file}")
        print("请先运行: python convert_annotations_to_coco.py")
        return False
    
    # 读取数据集信息
    with open(dataset_info_file, 'r') as f:
        dataset_info = json.load(f)
    
    print(f"✅ 读取数据集信息:")
    print(f"  数据根目录: {dataset_info['data_root']}")
    print(f"  类别数量: {dataset_info['num_classes']}")
    print(f"  训练标注: {dataset_info['ann_file']['train']}")
    print(f"  验证标注: {dataset_info['ann_file']['val']}")
    
    # 创建新的配置文件
    config_content = f'''# Mask R-CNN配置文件 - 使用转换后的COCO格式数据
# 基于VOC2012数据集，支持Tensorboard可视化

# 导入自定义钩子
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_hooks import CustomLoggerHook, TensorboardLoggerHook

# 模型配置
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes={dataset_info['num_classes']},
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes={dataset_info['num_classes']},
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # 训练配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    # 测试配置
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# 数据集配置
dataset_type = 'CocoDataset'
data_root = '{os.path.abspath(coco_data_root)}'

# 数据处理管道
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{dataset_info['ann_file']['train']}',
        data_prefix=dict(img='{dataset_info['img_prefix']}'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{dataset_info['ann_file']['val']}',
        data_prefix=dict(img='{dataset_info['img_prefix']}'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/{dataset_info['ann_file']['val']}',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[30, 37],
        gamma=0.1)
]

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        save_best='auto',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 自定义钩子配置
custom_hooks = [
    CustomLoggerHook(interval=50),
    TensorboardLoggerHook(interval=50)
]

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='work_dirs/mask_rcnn_coco/tensorboard_logs'
    )
]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# 类别名称（用于可视化）
metainfo = dict(
    classes={tuple(dataset_info['classes'])},
    palette=None
)
'''

    # 保存配置文件
    config_file = "configs/mask_rcnn_coco.py"
    Path("configs").mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Mask R-CNN配置文件已更新: {config_file}")
    
    return config_file


def update_train_script():
    """更新训练脚本以支持新的配置"""
    
    # 读取现有的训练脚本
    train_script = "train_models.py"
    
    if not os.path.exists(train_script):
        print(f"❌ 训练脚本不存在: {train_script}")
        return False
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原始文件
    backup_file = train_script + '.backup_coco'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 修改配置文件路径
    updated_content = content.replace(
        'config_file = "configs/mask_rcnn_voc.py"',
        'config_file = "configs/mask_rcnn_coco.py"'
    )
    
    # 更新工作目录
    updated_content = updated_content.replace(
        'work_dir = "work_dirs/mask_rcnn"',
        'work_dir = "work_dirs/mask_rcnn_coco"'
    )
    
    # 更新测试函数中的配置文件路径
    updated_content = updated_content.replace(
        'test_model("configs/mask_rcnn_voc.py"',
        'test_model("configs/mask_rcnn_coco.py"'
    )
    
    # 保存更新后的文件
    with open(train_script, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ 训练脚本已更新: {train_script}")
    print(f"📋 备份文件: {backup_file}")
    
    return True


def verify_setup():
    """验证设置是否正确"""
    print("\n🔍 验证设置...")
    
    # 检查必要文件
    required_files = [
        "./data/coco_format/annotations/instances_train2012.json",
        "./data/coco_format/annotations/instances_val2012.json",
        "./data/coco_format/dataset_info.json",
        "configs/mask_rcnn_coco.py",
        "custom_hooks.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # 检查图像目录
    images_dir = "./data/coco_format/images"
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        print(f"✅ 图像目录: {images_dir} ({image_count} 张图像)")
    else:
        print(f"❌ 图像目录不存在: {images_dir}")
        return False
    
    # 检查标注文件
    train_ann = "./data/coco_format/annotations/instances_train2012.json"
    val_ann = "./data/coco_format/annotations/instances_val2012.json"
    
    try:
        with open(train_ann, 'r') as f:
            train_data = json.load(f)
        with open(val_ann, 'r') as f:
            val_data = json.load(f)
        
        print(f"✅ 训练集标注: {len(train_data['images'])} 张图像, {len(train_data['annotations'])} 个标注")
        print(f"✅ 验证集标注: {len(val_data['images'])} 张图像, {len(val_data['annotations'])} 个标注")
        
    except Exception as e:
        print(f"❌ 标注文件验证失败: {e}")
        return False
    
    print("✅ 所有设置验证通过!")
    return True


def main():
    """主函数"""
    print("🔧 更新MMDetection配置以使用COCO格式数据")
    print("=" * 60)
    
    # 1. 更新Mask R-CNN配置
    config_file = update_mask_rcnn_config()
    if not config_file:
        return
    
    # 2. 更新训练脚本
    if not update_train_script():
        return
    
    # 3. 验证设置
    if not verify_setup():
        return
    
    print("\n" + "=" * 60)
    print("🎉 配置更新完成!")
    print("=" * 60)
    print("📋 更新内容:")
    print("  ✅ 创建了新的Mask R-CNN配置文件")
    print("  ✅ 更新了训练脚本")
    print("  ✅ 验证了数据和配置文件")
    
    print("\n💡 下一步:")
    print("1. 开始训练:")
    print("   python train_models.py --model mask_rcnn")
    print()
    print("2. 启动Tensorboard监控:")
    print("   python start_tensorboard.py")
    print()
    print("3. 在浏览器中查看训练进度:")
    print("   http://localhost:6006")
    
    print("\n📊 预期的Tensorboard可视化内容:")
    print("  🔸 Train_Loss/: 训练损失曲线")
    print("  🔸 Val_Loss/: 验证损失曲线")
    print("  🔸 Val_mAP/: 验证mAP曲线")
    print("  🔸 Learning_Rate/: 学习率变化")
    print("  🔸 Time/: 训练时间统计")


if __name__ == "__main__":
    main() 