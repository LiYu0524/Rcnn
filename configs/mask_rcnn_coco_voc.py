# Mask R-CNN配置文件 - 使用COCO格式的VOC2012数据集
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# 数据集设置
dataset_type = 'CocoDataset'
data_root = '/mnt/data/liyu/mm/data/coco/'

# VOC类别映射到COCO格式
metainfo = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157)
    ]
}

# 模型设置 - 禁用backbone的独立预训练权重
model = dict(
    backbone=dict(
        init_cfg=None  # 禁用backbone的独立预训练权重加载
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

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
                   'scale_factor')
    )
]

# 数据加载器设置 - 适配多卡训练
train_dataloader = dict(
    batch_size=8,  # 4卡训练时每卡2张图片，总批次大小8
    num_workers=4,  # 增加数据加载线程
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
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
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

test_dataloader = val_dataloader

# 评估器设置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator

# 训练设置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# 默认钩子设置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 自定义钩子 - 只使用CustomLoggerHook
custom_hooks = [
    dict(
        type='CustomLoggerHook',
        interval=50,
        log_metric_by_epoch=True,
        log_train_loss=True,
        log_val_loss=True,
        log_val_map=True
    )
]

# 可视化设置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='./work_dirs/mask_rcnn_coco_voc/tensorboard'
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录
work_dir = './work_dirs/mask_rcnn_coco_voc'

# 使用本地预训练权重 - 包含完整的Mask R-CNN模型权重
load_from = '/mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

# 恢复训练
resume = False

# 随机种子
randomness = dict(seed=0, deterministic=False)

# 自动缩放学习率配置 - 多卡训练时自动调整学习率
auto_scale_lr = dict(enable=True, base_batch_size=16) 