#!/usr/bin/env python3
"""
训练Mask R-CNN模型 - 使用COCO格式的VOC2012数据集
包含Tensorboard可视化和分离的loss记录
"""

import os
import sys
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

# 导入自定义钩子
import custom_hooks

def main():
    parser = argparse.ArgumentParser(description='训练Mask R-CNN模型')
    parser.add_argument('--config', 
                       default='configs/mask_rcnn_coco_voc.py',
                       help='配置文件路径')
    parser.add_argument('--work-dir', 
                       default='./work_dirs/mask_rcnn_coco_voc',
                       help='工作目录')
    parser.add_argument('--resume', 
                       action='store_true',
                       help='是否恢复训练')
    parser.add_argument('--amp', 
                       action='store_true',
                       help='是否使用自动混合精度')
    
    args = parser.parse_args()
    
    # 注册所有模块
    register_all_modules()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    cfg.work_dir = args.work_dir
    
    # 设置恢复训练
    if args.resume:
        cfg.resume = True
    
    # 设置自动混合精度
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    
    # 确保自定义钩子路径正确
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 创建runner并开始训练
    runner = Runner.from_cfg(cfg)
    
    print("=" * 50)
    print("开始训练Mask R-CNN模型")
    print(f"配置文件: {args.config}")
    print(f"工作目录: {args.work_dir}")
    print(f"训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"数据集: COCO格式的VOC2012")
    print(f"类别数: 20")
    print("=" * 50)
    
    # 开始训练
    runner.train()
    
    print("=" * 50)
    print("训练完成！")
    print(f"模型保存在: {args.work_dir}")
    print(f"Tensorboard日志: {args.work_dir}/tensorboard")
    print("=" * 50)

if __name__ == '__main__':
    main() 