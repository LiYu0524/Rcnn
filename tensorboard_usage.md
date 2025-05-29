# 📊 Tensorboard 可视化使用指南

## 🚀 快速启动

### 1. 查看最新训练的Tensorboard
```bash
# 自动查找并启动最新训练的Tensorboard
python view_latest_tensorboard.py
```

### 2. 列出所有可用的训练日志
```bash
# 只查看有哪些训练日志，不启动Tensorboard
python view_latest_tensorboard.py --list
```

### 3. 自定义端口启动
```bash
# 使用自定义端口（如果6006端口被占用）
python view_latest_tensorboard.py --port 6007
```

### 4. 启动通用Tensorboard
```bash
# 启动原始的Tensorboard脚本
python start_tensorboard.py
```

## 🌐 访问方式

### 本地访问
- 浏览器打开：`http://localhost:6006`

### 远程访问（如果在服务器上）
- 浏览器打开：`http://服务器IP:6006`
- 例如：`http://10.176.42.44:6006`

## 📈 可视化内容

训练过程中会生成以下可视化内容：

### 📊 损失曲线 (Loss)
- **Train_Loss/**: 训练过程中的各种loss
  - `loss_cls`: 分类损失
  - `loss_bbox`: 边界框回归损失
  - `loss_iou`: IoU损失
  - `loss`: 总损失

### 📉 验证指标 (Validation)
- **Val_Loss/**: 验证过程中的损失
- **Val_mAP/**: 验证集的mAP指标
  - `mAP`: 平均精度均值
  - `mAP_50`: IoU=0.5时的mAP
  - `mAP_75`: IoU=0.75时的mAP

### 🔧 训练参数
- **Learning_Rate/**: 学习率变化曲线
- **Time/**: 训练和验证时间统计

## 🛠️ 故障排除

### 1. 端口被占用
```bash
# 使用不同端口
python view_latest_tensorboard.py --port 6007
```

### 2. 未找到训练日志
- 确保训练已经开始
- 检查 `work_dirs/` 目录是否存在
- 确保配置文件中包含了Tensorboard配置

### 3. Tensorboard未安装
```bash
pip install tensorboard
```

### 4. 远程访问问题
- 确保服务器防火墙允许对应端口
- 使用 `--host 0.0.0.0` 参数

## 📝 训练日志位置

训练日志通常保存在以下位置：
- `work_dirs/sparse_rcnn/`: Sparse R-CNN训练日志
- `work_dirs/mask_rcnn/`: Mask R-CNN训练日志
- `work_dirs/sparse_rcnn/tensorboard_logs/`: 专用Tensorboard日志（如果配置了）

## 🎯 最佳实践

1. **实时监控**: 训练开始后立即启动Tensorboard
2. **多实验对比**: 可以同时查看多个实验的结果
3. **定期检查**: 通过loss曲线判断训练是否正常
4. **早停策略**: 根据验证mAP曲线决定是否提前停止训练

## 📞 常用命令总结

```bash
# 查看最新训练
python view_latest_tensorboard.py

# 列出所有训练日志
python view_latest_tensorboard.py --list

# 自定义端口
python view_latest_tensorboard.py --port 6007

# 后台运行
nohup python view_latest_tensorboard.py > tensorboard.log 2>&1 &
``` 