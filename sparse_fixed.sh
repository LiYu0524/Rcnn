#!/bin/bash
# 修复版Sparse R-CNN训练脚本
# 使用简化配置文件避免继承语法问题

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train_models_fixed.py --model sparse_rcnn --no-interactive > sparse_tensor_fixed.log 2>&1 &
 
echo "✅ Sparse R-CNN训练已启动（修复版）"
echo "📁 使用配置文件: sparse_rcnn_config_simple.py"
echo "📝 日志文件: sparse_tensor_fixed.log"
echo "🔍 查看日志: tail -f sparse_tensor_fixed.log" 