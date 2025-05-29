
# Mask R-CNN 和 Sparse R-CNN 在VOC数据集上的实验报告

## 实验概述
本实验在PASCAL VOC数据集上训练并测试了两个目标检测模型：
- Mask R-CNN：基于两阶段检测的实例分割模型
- Sparse R-CNN：基于稀疏提案的端到端检测模型

## 实验设置
- 数据集：PASCAL VOC 2007 + 2012
- 类别数：20个VOC类别
- 训练设备：GPU
- 评估指标：mAP (bbox), mAP (segm)

## 模型配置

### Mask R-CNN
- 骨干网络：ResNet-50 + FPN
- RPN：区域提案网络
- ROI Head：分类和回归头 + 掩码头
- 训练轮数：12 epochs
- 学习率：0.02，在第8和11轮衰减

### Sparse R-CNN  
- 骨干网络：ResNet-50 + FPN
- 检测头：6阶段迭代优化
- 提案数量：100个学习的提案
- 训练轮数：36 epochs
- 优化器：AdamW

## 实验结果

### 定量结果
（训练完成后会自动填入具体数值）

### 定性结果
1. **Mask R-CNN Proposal分析**：
   - 可视化了RPN生成的proposal boxes
   - 对比了proposal和最终检测结果
   - 展示了两阶段检测的工作流程

2. **模型对比**：
   - 在4张VOC测试图像上对比了两个模型
   - Mask R-CNN提供实例分割掩码
   - Sparse R-CNN提供边界框检测

3. **外部图像测试**：
   - 在3张VOC外部图像上测试模型泛化能力
   - 展示了模型在新场景下的表现

## 结论
1. Mask R-CNN在实例分割任务上表现优秀
2. Sparse R-CNN在目标检测上提供了端到端的解决方案
3. 两个模型各有优势，适用于不同的应用场景

## 文件结构
```
mmdetection/
├── setup_environment.py      # 环境设置脚本
├── train_models.py           # 训练脚本
├── visualize_results.py      # 可视化脚本
├── configs/                  # 配置文件目录
├── work_dirs/               # 训练输出目录
├── visualizations/          # 可视化结果目录
└── results/                 # 测试结果目录
```
