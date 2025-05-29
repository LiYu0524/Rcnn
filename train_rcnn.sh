#!/bin/bash
# echo "✅ Sparse R-CNN训练已启动"
# echo "📁 使用配置文件: sparse_rcnn_config_simple.py"
# echo "📝 日志文件: sparse_tensor_fixed.log"
# echo "🔍 查看日志: tail -f sparse_tensor_fixed.log"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train_models_fixed.py --model sparse_rcnn --no-interactive > sparse_tensor_fixed.log 2>&1 &


# echo "✅ Mask R-CNN训练已启动"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train_mask_rcnn_coco.py \
    --config configs/mask_rcnn_coco_voc.py \
    --work-dir work_dirs/mask_rcnn_coco_voc \
    > train_mask_rcnn_coco.log 2>&1 &