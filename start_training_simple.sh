#!/bin/bash

# 简化的训练启动脚本 - 测试单卡训练和Tensorboard

echo "=========================================="
echo "启动Mask R-CNN训练 - COCO格式VOC2012数据集"
echo "=========================================="

# 设置环境变量
export PYTHONPATH=$PWD:$PYTHONPATH

# 配置参数
CONFIG="configs/mask_rcnn_coco_voc.py"
WORK_DIR="work_dirs/mask_rcnn_coco_voc"
TENSORBOARD_PORT=6006

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmm

# 创建工作目录
mkdir -p ${WORK_DIR}/tensorboard

# 测试配置文件
echo "测试配置文件..."
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
sleep 3

# 显示训练信息
echo "=========================================="
echo "训练配置信息:"
echo "配置文件: ${CONFIG}"
echo "工作目录: ${WORK_DIR}"
echo "Tensorboard端口: ${TENSORBOARD_PORT}"
echo "=========================================="

# 开始单卡训练
echo "开始单卡训练..."
python train_mask_rcnn_coco.py \
    --config ${CONFIG} \
    --work-dir ${WORK_DIR} \
    > train_single_gpu.log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，PID: ${TRAIN_PID}"

# 监控训练进程
echo "监控训练进程..."
echo "查看训练日志: tail -f train_single_gpu.log"
echo "查看Tensorboard日志: tail -f tensorboard.log"
echo ""
echo "停止训练: kill ${TRAIN_PID}"
echo "停止Tensorboard: kill ${TENSORBOARD_PID}"
echo ""

# 等待几秒后显示初始日志
sleep 10
echo "显示训练初始日志:"
tail -20 train_single_gpu.log

echo ""
echo "继续查看实时日志请运行: tail -f train_single_gpu.log" 