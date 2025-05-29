#!/usr/bin/env python3
"""
测试简化的预训练配置文件
"""

import sys
import os
sys.path.append('mmdetection_repo')

try:
    from mmengine.config import Config
    
    print("🧪 测试简化的预训练配置文件...")
    cfg = Config.fromfile('configs/mask_rcnn_voc_pretrained_simple.py')
    
    print("✅ 简化配置文件加载成功!")
    print(f"📦 预训练权重: {cfg.model.init_cfg.checkpoint}")
    print(f"🎯 边界框类别数: {cfg.model.roi_head.bbox_head.num_classes}")
    print(f"🎭 掩码类别数: {cfg.model.roi_head.mask_head.num_classes}")
    print(f"📈 学习率: {cfg.optim_wrapper.optimizer.lr}")
    print(f"🔄 最大轮次: {cfg.train_cfg.max_epochs}")
    print(f"📚 训练集: {cfg.train_dataloader.dataset.ann_file}")
    print(f"📚 验证集: {cfg.val_dataloader.dataset.ann_file}")
    
    # 检查分层学习率
    if hasattr(cfg.optim_wrapper, 'paramwise_cfg'):
        print("✅ 分层学习率配置:")
        custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
        for key, config in custom_keys.items():
            lr_mult = config.lr_mult
            print(f"   {key}: {lr_mult}x")
    
    print("\n🎉 配置文件测试通过!")
    
except Exception as e:
    print(f"❌ 配置文件测试失败: {e}")
    import traceback
    traceback.print_exc() 