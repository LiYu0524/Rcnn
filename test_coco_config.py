#!/usr/bin/env python3
"""
测试COCO格式配置文件是否正确
"""

import os
import sys
from mmengine.config import Config
from mmdet.utils import register_all_modules

def test_config():
    # 注册所有模块
    register_all_modules()
    
    # 配置文件路径
    config_path = 'configs/mask_rcnn_coco_voc.py'
    
    try:
        # 加载配置
        cfg = Config.fromfile(config_path)
        print(f"✓ 配置文件加载成功: {config_path}")
        
        # 检查数据集路径
        data_root = cfg.data_root
        print(f"✓ 数据集路径: {data_root}")
        
        # 检查训练数据
        train_ann = os.path.join(data_root, cfg.train_dataloader.dataset.ann_file)
        train_img = os.path.join(data_root, cfg.train_dataloader.dataset.data_prefix.img)
        
        if os.path.exists(train_ann):
            print(f"✓ 训练标注文件存在: {train_ann}")
        else:
            print(f"✗ 训练标注文件不存在: {train_ann}")
            
        if os.path.exists(train_img):
            print(f"✓ 训练图片目录存在: {train_img}")
        else:
            print(f"✗ 训练图片目录不存在: {train_img}")
        
        # 检查验证数据
        val_ann = os.path.join(data_root, cfg.val_dataloader.dataset.ann_file)
        val_img = os.path.join(data_root, cfg.val_dataloader.dataset.data_prefix.img)
        
        if os.path.exists(val_ann):
            print(f"✓ 验证标注文件存在: {val_ann}")
        else:
            print(f"✗ 验证标注文件不存在: {val_ann}")
            
        if os.path.exists(val_img):
            print(f"✓ 验证图片目录存在: {val_img}")
        else:
            print(f"✗ 验证图片目录不存在: {val_img}")
        
        # 检查配置参数
        print(f"✓ 训练轮数: {cfg.train_cfg.max_epochs}")
        print(f"✓ 验证间隔: {cfg.train_cfg.val_interval}")
        print(f"✓ 类别数: {cfg.model.roi_head.bbox_head.num_classes}")
        print(f"✓ 批次大小: {cfg.train_dataloader.batch_size}")
        print(f"✓ 工作目录: {cfg.work_dir}")
        
        print("\n配置文件测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

if __name__ == '__main__':
    test_config() 