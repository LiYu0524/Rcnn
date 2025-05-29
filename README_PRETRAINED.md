# Mask R-CNN 预训练模型微调指南

## 📋 概述

本指南介绍如何使用COCO预训练的Mask R-CNN模型在VOC2012数据集上进行微调，实现更好的性能和更快的收敛。

## 🎯 核心特性

### 预训练策略
- **完整模型预训练**: 使用COCO训练的完整Mask R-CNN模型
- **智能权重初始化**: 重新初始化分类头和掩码头以适应VOC类别数
- **分层学习率**: 不同网络层使用不同的学习率
- **微调优化**: 针对预训练模型优化的训练策略

### 可视化功能
- **分离的Tensorboard日志**: 训练loss、验证loss、验证mAP分别记录
- **学习率监控**: 监控不同层的学习率变化
- **性能对比**: 与从头训练的模型进行对比

## 🚀 快速开始

### 1. 环境检查

```bash
# 测试配置文件和依赖项
python test_pretrained_config.py
```

### 2. 检查预训练权重

```bash
# 检查预训练权重是否存在
python train_pretrained_mask_rcnn.py --check-weights
```

### 3. 开始训练

```bash
# 训练并测试模型
python train_pretrained_mask_rcnn.py

# 或者分步执行
python train_pretrained_mask_rcnn.py --train    # 只训练
python train_pretrained_mask_rcnn.py --test     # 只测试
```

## 📊 配置详解

### 模型配置 (`configs/mask_rcnn_voc_pretrained.py`)

```python
# 使用完整的COCO预训练模型
model = dict(
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    ),
    
    # 重新初始化分类头和掩码头
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,  # VOC类别数
            init_cfg=dict(type='Normal', ...)  # 重新初始化
        ),
        mask_head=dict(
            num_classes=20,  # VOC类别数
            init_cfg=dict(type='Normal', ...)  # 重新初始化
        )
    )
)
```

### 优化器配置

```python
# 分层学习率策略
optim_wrapper = dict(
    optimizer=dict(lr=0.002),  # 微调基础学习率
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),   # backbone: 0.0002
            'neck': dict(lr_mult=0.5),       # neck: 0.001
            'rpn_head': dict(lr_mult=1.0),   # RPN: 0.002
            'roi_head': dict(lr_mult=1.0)    # ROI: 0.002
        }
    )
)
```

### 学习率调度

```python
# 微调学习率调度
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, end=500),  # 预热
    dict(type='MultiStepLR', milestones=[25, 35], gamma=0.1)  # 主训练
]
```

## 📈 性能对比

### 预期效果

| 训练方式 | 收敛速度 | 最终mAP | 训练时间 |
|---------|---------|---------|----------|
| 从头训练 | 慢 | ~65% | 40 epochs |
| 预训练微调 | 快 | ~70%+ | 25-30 epochs |

### 优势分析

1. **更快收敛**: 预训练模型已经学习了丰富的特征表示
2. **更高精度**: COCO数据集的知识迁移到VOC任务
3. **更稳定训练**: 避免从随机初始化开始的不稳定性
4. **更少资源**: 可能在更少的epoch内达到更好效果

## 🔧 高级配置

### GPU数量自适应

脚本会自动检测GPU数量并调整学习率：

```python
# 单GPU: lr = 0.002
# 双GPU: lr = 0.004
# 四GPU: lr = 0.008
```

### 数据增强

针对微调优化的数据增强策略：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',  # 增强数据多样性
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='PackDetInputs')
]
```

## 📊 Tensorboard可视化

### 启动Tensorboard

```bash
tensorboard --logdir=work_dirs/mask_rcnn_pretrained/tensorboard_logs --port=6006
```

### 可视化内容

1. **Train_Loss/**: 训练过程中的各种loss
   - `loss_cls`: 分类损失
   - `loss_bbox`: 边界框回归损失
   - `loss_mask`: 掩码损失
   - `loss_rpn_cls`: RPN分类损失
   - `loss_rpn_bbox`: RPN回归损失

2. **Val_Loss/**: 验证过程中的各种loss

3. **Val_mAP/**: 验证集性能指标
   - `bbox_mAP`: 边界框mAP
   - `segm_mAP`: 分割mAP
   - 各类别AP值

4. **Learning_Rate/**: 分层学习率变化
   - `backbone_lr`: Backbone学习率
   - `neck_lr`: Neck学习率
   - `head_lr`: 检测头学习率

## 📁 输出文件结构

```
work_dirs/mask_rcnn_pretrained/
├── tensorboard_logs/           # Tensorboard日志
├── best_auto_*.pth            # 最佳模型权重
├── latest.pth                 # 最新模型权重
├── epoch_*.pth               # 各epoch权重
└── *.log                     # 训练日志

results/
└── mask_rcnn_pretrained_final_results.pkl  # 测试结果

visualizations/mask_rcnn_pretrained/
├── *.jpg                     # 可视化结果图像
└── ...
```

## 🔍 故障排除

### 常见问题

1. **预训练权重不存在**
   ```bash
   # 复制权重文件
   mkdir -p pretrain
   cp /mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth pretrain/
   ```

2. **配置文件加载失败**
   ```bash
   # 测试配置文件
   python test_pretrained_config.py
   ```

3. **GPU内存不足**
   - 减小batch_size (默认为2)
   - 使用梯度累积
   - 使用混合精度训练

4. **学习率过高/过低**
   - 检查GPU数量自适应是否正确
   - 手动调整配置文件中的学习率

### 调试技巧

1. **监控训练过程**
   ```bash
   # 实时查看训练日志
   tail -f work_dirs/mask_rcnn_pretrained/*.log
   ```

2. **检查模型权重加载**
   - 查看训练开始时的日志输出
   - 确认哪些权重被重新初始化

3. **验证数据加载**
   - 检查数据集路径是否正确
   - 确认train.txt和val.txt文件存在

## 💡 最佳实践

### 训练策略

1. **学习率选择**: 从0.002开始，根据收敛情况调整
2. **早停策略**: 监控验证mAP，连续5个epoch无提升时考虑停止
3. **权重保存**: 只保留最佳模型和最近3个检查点
4. **数据增强**: 适度使用，避免过度增强影响收敛

### 性能优化

1. **批次大小**: 根据GPU内存调整，推荐2-4
2. **工作进程**: num_workers设置为CPU核心数的1/4
3. **混合精度**: 可以考虑使用fp16加速训练
4. **分布式训练**: 多GPU时使用分布式训练

## 📚 参考资料

- [MMDetection官方文档](https://mmdetection.readthedocs.io/)
- [Mask R-CNN论文](https://arxiv.org/abs/1703.06870)
- [VOC数据集说明](http://host.robots.ox.ac.uk/pascal/VOC/)
- [迁移学习最佳实践](https://cs231n.github.io/transfer-learning/)

## 🤝 贡献

如果您发现问题或有改进建议，请：
1. 检查现有的issue
2. 创建新的issue描述问题
3. 提交pull request

---

**注意**: 本配置针对VOC2012数据集优化，如需适配其他数据集，请相应调整类别数和数据路径。 