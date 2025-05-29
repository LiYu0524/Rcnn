#!/bin/bash

# 多卡训练Mask R-CNN模型 - 使用COCO格式的VOC2012数据集
# 包含Tensorboard可视化和分离的loss记录

echo "=========================================="
echo "多卡训练Mask R-CNN模型 - COCO格式VOC2012数据集"
echo "=========================================="

# 设置环境变量
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用前4张GPU

# 配置参数
GPUS=4
CONFIG="configs/mask_rcnn_coco_voc.py"
WORK_DIR="work_dirs/mask_rcnn_coco_voc"
PORT=29500
TENSORBOARD_PORT=6006

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

# 检查预训练权重
if [ ! -f "/mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth" ]; then
    echo "错误: 预训练权重文件不存在"
    exit 1
fi

echo "预训练权重检查通过！"

# 创建工作目录
mkdir -p ${WORK_DIR}/tensorboard

# 测试配置文件
echo "测试配置文件..."
conda activate openmm
python test_coco_config.py
if [ $? -ne 0 ]; then
    echo "配置文件测试失败，请检查配置"
    exit 1
fi

# 启动Tensorboard
echo "启动Tensorboard..."
tensorboard --logdir=${WORK_DIR}/tensorboard --port=${TENSORBOARD_PORT} --host=0.0.0.0 > tensorboard.log 2>&1 &
TENSORBOARD_PID=$!
echo "Tensorboard已启动，PID: ${TENSORBOARD_PID}"
echo "Tensorboard访问地址: http://localhost:${TENSORBOARD_PORT}"

# 等待Tensorboard启动
sleep 5

# 显示训练信息
echo "=========================================="
echo "训练配置信息:"
echo "GPU数量: ${GPUS}"
echo "配置文件: ${CONFIG}"
echo "工作目录: ${WORK_DIR}"
echo "分布式端口: ${PORT}"
echo "Tensorboard端口: ${TENSORBOARD_PORT}"
echo "=========================================="

# 开始多卡训练
echo "开始多卡训练..."
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    mmdetection_repo/tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR} \
    > train_multi_gpu.log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，PID: ${TRAIN_PID}"

# 监控训练进程
echo "监控训练进程..."
echo "查看训练日志: tail -f train_multi_gpu.log"
echo "查看Tensorboard日志: tail -f tensorboard.log"
echo ""
echo "停止训练: kill ${TRAIN_PID}"
echo "停止Tensorboard: kill ${TENSORBOARD_PID}"
echo ""

# 等待用户输入
echo "按任意键查看训练日志，或按Ctrl+C退出..."
read -n 1

# 显示训练日志
tail -f train_multi_gpu.log 