# Mask R-CNN配置文件 - VOC2012数据集（实例分割）

# 导入自定义钩子
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_hooks import CustomLoggerHook, TensorboardLoggerHook

_base_ = [
    '../mmdetection_repo/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../mmdetection_repo/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection_repo/configs/_base_/default_runtime.py'
]

# 模型配置 - 使用本地预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/resnet50-0676ba61.pth')
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=20),  # VOC有20个类别
        mask_head=dict(num_classes=20)   # 实例分割也是20个类别
    )
)

# 数据集配置 - 使用VOC2012
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

# 数据处理管道 - 包含掩码处理
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 数据加载器配置 - 使用VOC2012
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2012/ImageSets/Main/train.txt',  # 使用VOC2012训练集
        data_prefix=dict(sub_data_root='VOC2012/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2012/ImageSets/Main/val.txt',  # 使用VOC2012验证集
        data_prefix=dict(sub_data_root='VOC2012/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

test_dataloader = val_dataloader

# 评估器配置 - 评估边界框和分割掩码
val_evaluator = dict(type='VOCMetric', metric=['mAP'], eval_mode='11points')
test_evaluator = val_evaluator

# 优化器配置 - 适配8卡分布式训练，40个epoch
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02 * 8, momentum=0.9, weight_decay=0.0001)  # 8卡训练，学习率线性缩放
)

# 学习率调度 - 适配40个epoch
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,  # 训练40个epoch
        by_epoch=True,
        milestones=[30, 37],  # 在30和37个epoch时降低学习率
        gamma=0.1)
]

# 训练配置 - 40个epoch，每个epoch验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认钩子 - 基于验证mAP保存最佳模型
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='auto',  # 自动保存最佳模型
        rule='greater'  # mAP越大越好
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 可视化配置 - 添加分离的Tensorboard支持
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='work_dirs/mask_rcnn/tensorboard_logs'
    )
]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# 自定义钩子 - 分离记录训练loss、验证loss、验证mAP
custom_hooks = [
    dict(
        type='CustomLoggerHook',
        log_metric_by_epoch=True,
        log_train_loss=True,
        log_val_loss=True,
        log_val_map=True
    )
]

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
