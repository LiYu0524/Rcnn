# Mask R-CNN 和 Sparse R-CNN 在VOC数据集上的训练与测试

本项目实现了在PASCAL VOC数据集上训练和测试Mask R-CNN和Sparse R-CNN模型的完整流程。

## 项目概述

本项目满足以下要求：
1. 使用MMDetection框架在VOC数据集上训练Mask R-CNN和Sparse R-CNN
2. 可视化对比Mask R-CNN的proposal box和最终预测结果
3. 对比两个模型的实例分割与目标检测结果
4. 在外部图像上测试模型的泛化能力

## 项目结构

```
mmdetection/
├── README.md                    # 项目说明文档
├── run_experiment.py           # 主运行脚本
├── setup_environment.py       # 环境设置脚本
├── train_models.py            # 模型训练脚本
├── visualize_results.py       # 结果可视化脚本
├── sparse_rcnn_config.py      # Sparse R-CNN配置文件
├── configs/                   # 配置文件目录
│   └── mask_rcnn_voc.py      # Mask R-CNN配置文件
├── data/                      # 数据集目录
│   └── VOCdevkit/            # VOC数据集
├── work_dirs/                 # 训练输出目录
│   ├── mask_rcnn/            # Mask R-CNN训练结果
│   └── sparse_rcnn/          # Sparse R-CNN训练结果
├── visualizations/            # 可视化结果目录
├── results/                   # 测试结果目录
└── external_images/           # 外部测试图像
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA（推荐，用于GPU训练）
- 足够的存储空间（VOC数据集约2GB，训练模型约1GB）

## 快速开始

### 方法1：一键运行（推荐）

```bash
# 进入项目目录
cd mmdetection

# 运行完整实验（包括环境设置、训练、测试、可视化）
python run_experiment.py

# 或者只训练特定模型
python run_experiment.py --model mask_rcnn
python run_experiment.py --model sparse_rcnn
```

### 方法2：分步执行

#### 步骤1：环境设置
```bash
python setup_environment.py
```
这个脚本会：
- 安装MMDetection及其依赖
- 下载VOC 2007和2012数据集
- 创建必要的配置文件

#### 步骤2：训练模型
```bash
# 训练两个模型
python train_models.py --model both

# 或者单独训练
python train_models.py --model mask_rcnn
python train_models.py --model sparse_rcnn
```

#### 步骤3：可视化结果
```bash
python visualize_results.py
```

## 详细功能说明

### 1. 环境设置 (`setup_environment.py`)

- **MMDetection安装**：自动安装MMDetection框架和所有依赖
- **数据集下载**：下载VOC2007和VOC2012数据集
- **配置文件生成**：创建适配VOC数据集的模型配置

### 2. 模型训练 (`train_models.py`)

#### Mask R-CNN配置
- 骨干网络：ResNet-50 + FPN
- 类别数：20（VOC类别）
- 训练轮数：12 epochs
- 学习率：0.02，在第8和11轮衰减
- 优化器：SGD

#### Sparse R-CNN配置
- 骨干网络：ResNet-50 + FPN
- 检测头：6阶段迭代优化
- 提案数量：100个学习的提案
- 训练轮数：36 epochs
- 优化器：AdamW

### 3. 结果可视化 (`visualize_results.py`)

#### 功能特性
- **Proposal可视化**：展示Mask R-CNN第一阶段生成的proposal boxes
- **结果对比**：对比最终检测结果与proposals
- **模型对比**：并排比较两个模型的检测结果
- **外部测试**：在VOC外部图像上测试模型

#### 输出内容
- 4张VOC测试图像的详细分析
- Mask R-CNN的proposal和最终结果对比
- 两个模型的检测结果对比
- 外部图像的检测结果

## 使用参数

### run_experiment.py 参数
```bash
python run_experiment.py [选项]

选项:
  --step {setup,train,visualize,report,all}
                        选择要执行的步骤 (默认: all)
  --model {mask_rcnn,sparse_rcnn,both}
                        选择要训练的模型 (默认: both)
  --skip-setup         跳过环境设置步骤
  --skip-train         跳过训练步骤（使用已有模型）
```

### train_models.py 参数
```bash
python train_models.py [选项]

选项:
  --model {mask_rcnn,sparse_rcnn,both}
                        选择要训练的模型 (默认: both)
  --test-only          只进行测试，不训练
```

### visualize_results.py 参数
```bash
python visualize_results.py [选项]

选项:
  --mask-rcnn-config   Mask R-CNN配置文件路径
  --mask-rcnn-checkpoint Mask R-CNN检查点文件路径
  --sparse-rcnn-config Sparse R-CNN配置文件路径
  --sparse-rcnn-checkpoint Sparse R-CNN检查点文件路径
  --voc-test-dir       VOC测试图像目录
  --output-dir         输出目录
```

## 预期结果

### 训练输出
- `work_dirs/mask_rcnn/`：Mask R-CNN训练日志、配置和模型检查点
- `work_dirs/sparse_rcnn/`：Sparse R-CNN训练日志、配置和模型检查点

### 可视化输出
- `visualizations/mask_rcnn_proposals_*.png`：Mask R-CNN proposal分析
- `visualizations/model_comparison_*.png`：模型对比结果
- `visualizations/external_tests/`：外部图像测试结果

### 性能指标
- mAP (bbox)：边界框检测平均精度
- mAP (segm)：实例分割平均精度（仅Mask R-CNN）

## 常见问题

### Q: 训练时间需要多长？
A: 在单GPU上，Mask R-CNN约需要2-4小时，Sparse R-CNN约需要6-8小时。

### Q: 内存要求是多少？
A: 推荐至少8GB GPU内存。如果内存不足，可以减少batch size。

### Q: 如何使用预训练模型？
A: 可以使用`--skip-train`参数跳过训练，直接使用预训练模型进行测试。

### Q: 如何修改训练参数？
A: 编辑`configs/mask_rcnn_voc.py`和`sparse_rcnn_config.py`文件。

## 技术细节

### VOC数据集适配
- 将COCO格式的配置适配为VOC格式
- 类别数从80改为20
- 数据路径和标注格式调整

### 可视化技术
- 使用matplotlib进行结果可视化
- 支持边界框、类别标签、置信度分数显示
- 实例分割掩码可视化

### 模型对比
- 统一的评估指标
- 相同的测试图像
- 标准化的可视化格式

## 扩展功能

### 添加新模型
1. 在`configs/`目录下添加新的配置文件
2. 修改`train_models.py`添加训练逻辑
3. 更新`visualize_results.py`支持新模型

### 自定义数据集
1. 修改配置文件中的数据路径
2. 调整类别数和类别名称
3. 更新数据加载pipeline

## 参考资料

- [MMDetection官方文档](https://mmdetection.readthedocs.io/)
- [PASCAL VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Mask R-CNN论文](https://arxiv.org/abs/1703.06870)
- [Sparse R-CNN论文](https://arxiv.org/abs/2011.12450)

## 许可证

本项目遵循MIT许可证。 