#!/bin/bash

# 训练Mask R-CNN模型 - 使用COCO格式的VOC2012数据集
# 包含Tensorboard可视化和分离的loss记录

echo "=========================================="
echo "训练Mask R-CNN模型 - COCO格式VOC2012数据集"
echo "=========================================="

# 设置环境变量
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查数据集
echo "检查数据集..."
if [ ! -d "/mnt/data/liyu/mm/data/coco/train2017" ]; then
    echo "错误: 训练图片目录不存在"
    exit 1
fi

if [ ! -d "/mnt/data/liyu/mm/data/coco/val2017" ]; then
    echo "错误: 验证图片目录不存在"
    exit 1
fi

if [ ! -f "/mnt/data/liyu/mm/data/coco/annotations/instances_train2017.json" ]; then
    echo "错误: 训练标注文件不存在"
    exit 1
fi

if [ ! -f "/mnt/data/liyu/mm/data/coco/annotations/instances_val2017.json" ]; then
    echo "错误: 验证标注文件不存在"
    exit 1
fi

echo "数据集检查通过！"

# 创建工作目录
mkdir -p work_dirs/mask_rcnn_coco_voc/tensorboard

# 测试配置文件
echo "测试配置文件..."
python test_coco_config.py
if [ $? -ne 0 ]; then
    echo "配置文件测试失败，请检查配置"
    exit 1
fi

# 开始训练
echo "开始训练..."
python train_mask_rcnn_coco.py \
    --config configs/mask_rcnn_coco_voc.py \
    --work-dir work_dirs/mask_rcnn_coco_voc \
    > train_mask_rcnn_coco.log 2>&1

echo "=========================================="
echo "训练完成！"
echo "模型保存在: work_dirs/mask_rcnn_coco_voc"
echo "Tensorboard日志: work_dirs/mask_rcnn_coco_voc/tensorboard"
echo ""
echo "查看Tensorboard:"
echo "tensorboard --logdir=work_dirs/mask_rcnn_coco_voc/tensorboard --port=6006"
echo "==========================================" 