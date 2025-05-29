#!/usr/bin/env python3
"""
多卡训练Mask R-CNN模型 - 使用COCO格式的VOC2012数据集
包含Tensorboard可视化和分离的loss记录
"""

import os
import sys
import argparse
import subprocess
from mmengine.config import Config
from mmdet.utils import register_all_modules

# 导入自定义钩子
import custom_hooks

def main():
    parser = argparse.ArgumentParser(description='多卡训练Mask R-CNN模型')
    parser.add_argument('--config', 
                       default='configs/mask_rcnn_coco_voc.py',
                       help='配置文件路径')
    parser.add_argument('--work-dir', 
                       default='./work_dirs/mask_rcnn_coco_voc',
                       help='工作目录')
    parser.add_argument('--gpus', 
                       type=int,
                       default=4,
                       help='使用的GPU数量')
    parser.add_argument('--port',
                       type=int,
                       default=29500,
                       help='分布式训练端口')
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
    
    # 多卡训练时调整批次大小
    if args.gpus > 1:
        # 总批次大小保持不变，每卡批次大小相应减少
        original_batch_size = cfg.train_dataloader.batch_size
        cfg.train_dataloader.batch_size = max(1, original_batch_size // args.gpus)
        
        # 调整学习率（线性缩放）
        cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr * args.gpus
    
    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'tensorboard'), exist_ok=True)
    
    # 保存修改后的配置
    config_save_path = os.path.join(args.work_dir, 'config.py')
    cfg.dump(config_save_path)
    
    print("=" * 60)
    print("开始多卡训练Mask R-CNN模型")
    print(f"配置文件: {args.config}")
    print(f"工作目录: {args.work_dir}")
    print(f"使用GPU数量: {args.gpus}")
    print(f"每卡批次大小: {cfg.train_dataloader.batch_size}")
    print(f"总批次大小: {cfg.train_dataloader.batch_size * args.gpus}")
    print(f"学习率: {cfg.optim_wrapper.optimizer.lr}")
    print(f"训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"数据集: COCO格式的VOC2012")
    print(f"类别数: 20")
    print("=" * 60)
    
    # 构建多卡训练命令
    if args.gpus == 1:
        # 单卡训练
        cmd = [
            'python', 'train_mask_rcnn_coco.py',
            '--config', config_save_path,
            '--work-dir', args.work_dir
        ]
    else:
        # 多卡训练
        cmd = [
            'python', '-m', 'torch.distributed.launch',
            f'--nproc_per_node={args.gpus}',
            f'--master_port={args.port}',
            'tools/train.py',  # 使用mmdetection的标准训练脚本
            config_save_path,
            '--launcher', 'pytorch'
        ]
        
        if args.resume:
            cmd.append('--resume')
        if args.amp:
            cmd.append('--amp')
    
    # 启动训练
    print("启动训练命令:")
    print(' '.join(cmd))
    print("=" * 60)
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
        print("=" * 60)
        print("训练完成！")
        print(f"模型保存在: {args.work_dir}")
        print(f"Tensorboard日志: {args.work_dir}/tensorboard")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 