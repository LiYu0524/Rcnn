# Tensorboard 可视化训练过程

## 🎯 功能概述

本项目已配置完整的Tensorboard可视化功能，支持分离记录和显示训练过程中的各种指标。

## 📋 数据集配置

- **训练集**: VOC2012 train.txt (5718张图片)
- **验证集**: VOC2012 val.txt (5824张图片)  
- **训练轮次**: 40个epoch
- **验证频率**: 每个epoch进行验证
- **模型保存**: 基于验证mAP保存最佳模型

## 🎨 分离的可视化内容

### 1. 训练Loss (Train_Loss/)
- `Train_Loss/loss`: 总训练loss
- `Train_Loss/loss_cls`: 分类loss
- `Train_Loss/loss_bbox`: 边界框回归loss
- `Train_Loss/loss_mask`: 掩码loss (仅Mask R-CNN)
- `Train_Loss/loss_rpn_cls`: RPN分类loss
- `Train_Loss/loss_rpn_bbox`: RPN回归loss

### 2. 验证Loss (Val_Loss/)
- `Val_Loss/val_loss`: 总验证loss
- `Val_Loss/val_loss_cls`: 验证分类loss
- `Val_Loss/val_loss_bbox`: 验证边界框回归loss
- `Val_Loss/val_loss_mask`: 验证掩码loss (仅Mask R-CNN)

### 3. 验证mAP (Val_mAP/)
- `Val_mAP/mAP`: 总体mAP
- `Val_mAP/mAP_50`: IoU=0.5时的mAP
- `Val_mAP/mAP_75`: IoU=0.75时的mAP
- `Val_mAP/mAP_s`: 小目标mAP
- `Val_mAP/mAP_m`: 中等目标mAP
- `Val_mAP/mAP_l`: 大目标mAP

### 4. 学习率 (Learning_Rate/)
- `Learning_Rate/lr_0`: 主学习率
- `Learning_Rate/lr_1`: 其他参数组学习率

### 5. 时间统计 (Time/)
- `Time/iter_time`: 每次迭代时间
- `Time/val_time`: 验证时间

### 6. 其他验证指标 (Val_Metrics/)
- 其他验证过程中的指标

## 🚀 使用方法

### 1. 启动训练
```bash
# 训练两个模型
python train_models.py --model both

# 只训练Mask R-CNN
python train_models.py --model mask_rcnn

# 只训练Sparse R-CNN  
python train_models.py --model sparse_rcnn

# 非交互模式（适合nohup后台运行）
python train_models.py --model both --no-interactive
```

### 2. 启动Tensorboard
训练开始后，在新的终端中运行：
```bash
# 使用脚本启动
python start_tensorboard.py

# 或手动启动
tensorboard --logdir=work_dirs --port=6006
```

### 3. 访问Tensorboard
在浏览器中访问: http://localhost:6006

### 4. 测试配置
在训练前可以测试配置文件是否正确：
```bash
python test_config.py
```

## 📊 Tensorboard界面说明

### Scalars标签页
- **Train_Loss**: 训练过程中的各种loss曲线
- **Val_Loss**: 验证过程中的各种loss曲线  
- **Val_mAP**: 验证集上的mAP指标曲线
- **Learning_Rate**: 学习率变化曲线
- **Time**: 训练和验证时间统计

### 查看技巧
1. 使用左侧的标签过滤器选择要查看的指标组
2. 可以同时选择多个模型进行对比
3. 使用平滑功能减少曲线噪声
4. 可以下载数据或图片

## 🏆 最佳模型选择

训练过程中会自动保存验证mAP最高的模型权重：
- 文件名格式: `best_auto_YYYYMMDD_HHMMSS.pth`
- 保存位置: `work_dirs/{model_name}/`

最终测试时会自动使用最佳权重进行评估。

## 📁 文件结构

```
work_dirs/
├── mask_rcnn/
│   ├── tensorboard_logs/          # Tensorboard日志
│   ├── best_auto_*.pth           # 最佳模型权重
│   └── latest.pth                # 最新模型权重
└── sparse_rcnn/
    ├── tensorboard_logs/          # Tensorboard日志  
    ├── best_auto_*.pth           # 最佳模型权重
    └── latest.pth                # 最新模型权重

results/
├── mask_rcnn_final_results.pkl   # 最终测试结果
└── sparse_rcnn_final_results.pkl # 最终测试结果

visualizations/
├── mask_rcnn/                    # 可视化结果
└── sparse_rcnn/                  # 可视化结果
```

## 🔧 自定义配置

如需修改可视化配置，可以编辑：
- `custom_hooks.py`: 自定义日志钩子
- `configs/mask_rcnn_voc.py`: Mask R-CNN配置
- `sparse_rcnn_config.py`: Sparse R-CNN配置

## ⚠️ 注意事项

1. 确保安装了tensorboard: `pip install tensorboard`
2. 训练开始后才会生成Tensorboard日志文件
3. 如果修改了配置文件，建议先运行`test_config.py`验证
4. 多卡训练时学习率会自动调整
5. 验证mAP用于保存最佳模型，确保验证集质量 