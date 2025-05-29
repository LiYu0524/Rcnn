#!/usr/bin/env python3
"""
测试配置文件加载的脚本
"""

import sys
import os

def test_config_loading():
    """测试配置文件是否能正常加载"""
    print("🔧 测试配置文件加载")
    print("="*50)
    
    try:
        # 测试使用MMEngine加载配置文件
        from mmengine.config import Config
        
        # 测试Sparse R-CNN配置
        print("测试 sparse_rcnn_config_simple.py...")
        sparse_cfg = Config.fromfile('sparse_rcnn_config_simple.py')
        print("✅ Sparse R-CNN配置加载成功")
        print(f"  模型类型: {sparse_cfg.model['type']}")
        print(f"  训练轮次: {sparse_cfg.train_cfg['max_epochs']}")
        print(f"  验证间隔: {sparse_cfg.train_cfg['val_interval']}")
        
        # 测试Mask R-CNN配置
        print("\n测试 configs/mask_rcnn_voc_simple.py...")
        mask_cfg = Config.fromfile('configs/mask_rcnn_voc_simple.py')
        print("✅ Mask R-CNN配置加载成功")
        print(f"  模型类型: {mask_cfg.model['type']}")
        print(f"  训练轮次: {mask_cfg.train_cfg['max_epochs']}")
        print(f"  验证间隔: {mask_cfg.train_cfg['val_interval']}")
        
        print("\n🎉 所有配置文件测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_loading() 