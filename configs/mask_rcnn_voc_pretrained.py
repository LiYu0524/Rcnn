# Mask R-CNN配置文件 - 使用COCO预训练模型微调VOC2012数据集

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

# 模型配置 - 使用完整的COCO预训练模型
model = dict(
    # 使用完整的COCO预训练权重
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    ),
    
    # 重新配置ROI头以适应VOC类别数
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,  # VOC有20个类别
            # 重新初始化分类头权重
            init_cfg=dict(
                type='Normal',
                layer=['Linear'],
                std=0.01,
                override=dict(
                    type='Normal',
                    name='fc_cls',
                    std=0.01,
                    bias_prob=0.01
                )
            )
        ),
        mask_head=dict(
            num_classes=20,  # VOC有20个类别
            # 重新初始化掩码头权重
            init_cfg=dict(
                type='Normal',
                layer=['Conv2d'],
                std=0.001,
                override=dict(
                    type='Normal',
                    name='conv_logits',
                    std=0.001
                )
            )
        )
    )
)

# 数据集配置 - 使用VOC2012
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

# 数据处理管道 - 针对微调优化
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # 添加数据增强以提高微调效果
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2012/ImageSets/Main/train.txt',
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
        ann_file='VOC2012/ImageSets/Main/val.txt',
        data_prefix=dict(sub_data_root='VOC2012/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(type='VOCMetric', metric=['mAP'], eval_mode='11points')
test_evaluator = val_evaluator

# 优化器配置 - 微调学习率策略
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.002,  # 微调学习率，比从头训练小10倍
        momentum=0.9, 
        weight_decay=0.0001
    ),
    # 分层学习率：backbone使用更小的学习率
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # backbone学习率再减小10倍
            'neck': dict(lr_mult=0.5),      # neck学习率减小2倍
            'rpn_head': dict(lr_mult=1.0),  # RPN头使用标准学习率
            'roi_head': dict(lr_mult=1.0)   # ROI头使用标准学习率
        }
    )
)

# 学习率调度 - 微调策略
param_scheduler = [
    # 预热阶段
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500
    ),
    # 主要训练阶段
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[25, 35],  # 微调时更晚降低学习率
        gamma=0.1
    )
]

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认钩子
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='auto',
        rule='greater',
        max_keep_ckpts=3  # 只保留最近3个检查点
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Tensorboard可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='work_dirs/mask_rcnn_pretrained/tensorboard_logs'
    )
]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

# 自定义钩子 - 分离记录不同类型的指标
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

# 工作目录
work_dir = 'work_dirs/mask_rcnn_pretrained' 