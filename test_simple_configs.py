#!/usr/bin/env python3
"""
测试简化配置文件的脚本
"""

import sys
import os

def test_simple_config():
    """测试简化配置文件"""
    print("🔧 测试简化配置文件")
    print("="*50)
    
    # 测试Sparse R-CNN配置
    try:
        print("测试 sparse_rcnn_config_simple.py...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # 直接导入配置文件作为模块
        import sparse_rcnn_config_simple as sparse_config
        
        print("✅ Sparse R-CNN配置加载成功")
        print(f"  模型类型: {sparse_config.model['type']}")
        print(f"  训练轮次: {sparse_config.train_cfg['max_epochs']}")
        print(f"  验证间隔: {sparse_config.train_cfg['val_interval']}")
        print(f"  优化器: {sparse_config.optim_wrapper['optimizer']['type']}")
        print(f"  学习率: {sparse_config.optim_wrapper['optimizer']['lr']}")
        
    except Exception as e:
        print(f"❌ Sparse R-CNN配置测试失败: {e}")
        return False
    
    print("-" * 50)
    
    # 测试Mask R-CNN配置
    try:
        print("测试 configs/mask_rcnn_voc_simple.py...")
        
        # 添加configs目录到路径
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
        
        # 直接导入配置文件作为模块
        import mask_rcnn_voc_simple as mask_config
        
        print("✅ Mask R-CNN配置加载成功")
        print(f"  模型类型: {mask_config.model['type']}")
        print(f"  训练轮次: {mask_config.train_cfg['max_epochs']}")
        print(f"  验证间隔: {mask_config.train_cfg['val_interval']}")
        print(f"  优化器: {mask_config.optim_wrapper['optimizer']['type']}")
        print(f"  学习率: {mask_config.optim_wrapper['optimizer']['lr']}")
        
    except Exception as e:
        print(f"❌ Mask R-CNN配置测试失败: {e}")
        return False
    
    print("-" * 50)
    print("🎉 所有简化配置文件测试通过!")
    print("\n💡 提示:")
    print("  - 配置文件已修复，避免了继承语法问题")
    print("  - 可以使用 train_models_fixed.py 进行训练")
    print("  - 或者使用 sparse_fixed.sh 启动Sparse R-CNN训练")
    
    return True

if __name__ == "__main__":
    test_simple_config() 