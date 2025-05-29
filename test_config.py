#!/usr/bin/env python3
"""
测试配置文件是否正确的脚本
"""

import sys
import os
from pathlib import Path

def test_config_file(config_file):
    """测试配置文件是否可以正确加载"""
    print(f"测试配置文件: {config_file}")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # 尝试导入mmdet
        from mmengine.config import Config
        
        # 加载配置
        cfg = Config.fromfile(config_file)
        print(f"✅ 配置文件加载成功")
        
        # 检查关键配置
        print(f"📋 数据集配置:")
        print(f"  训练集: {cfg.train_dataloader.dataset.ann_file}")
        print(f"  验证集: {cfg.val_dataloader.dataset.ann_file}")
        print(f"  训练轮次: {cfg.train_cfg.max_epochs}")
        print(f"  验证间隔: {cfg.train_cfg.val_interval}")
        
        # 检查模型类型
        print(f"📦 模型配置:")
        print(f"  模型类型: {cfg.model.type}")
        print(f"  类别数量: {cfg.model.roi_head.bbox_head.num_classes if hasattr(cfg.model.roi_head.bbox_head, 'num_classes') else cfg.model.roi_head.bbox_head[0].num_classes}")
        
        # 检查Tensorboard配置
        if hasattr(cfg, 'vis_backends'):
            tensorboard_backends = [b for b in cfg.vis_backends if b.type == 'TensorboardVisBackend']
            if tensorboard_backends:
                print(f"✅ Tensorboard配置正确")
                if 'save_dir' in tensorboard_backends[0]:
                    print(f"  日志目录: {tensorboard_backends[0].save_dir}")
            else:
                print(f"⚠️  未找到Tensorboard配置")
        
        # 检查自定义钩子
        if hasattr(cfg, 'custom_hooks'):
            custom_logger_hooks = [h for h in cfg.custom_hooks if h.type == 'CustomLoggerHook']
            if custom_logger_hooks:
                print(f"✅ 自定义日志钩子配置正确")
            else:
                print(f"⚠️  未找到自定义日志钩子")
        
        # 检查优化器配置
        print(f"🔧 优化器配置:")
        print(f"  优化器类型: {cfg.optim_wrapper.optimizer.type}")
        print(f"  学习率: {cfg.optim_wrapper.optimizer.lr}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 配置文件测试工具")
    print("="*50)
    
    # 测试新的简化配置文件
    configs = [
        "configs/mask_rcnn_voc_simple.py",
        "sparse_rcnn_config_simple.py"
    ]
    
    all_passed = True
    for config in configs:
        if os.path.exists(config):
            success = test_config_file(config)
            all_passed = all_passed and success
        else:
            print(f"❌ 配置文件不存在: {config}")
            all_passed = False
        print("-" * 50)
    
    if all_passed:
        print("🎉 所有配置文件测试通过!")
        print("\n💡 提示:")
        print("  - 新的简化配置文件已创建，避免了继承语法问题")
        print("  - 可以使用这些配置文件进行训练")
        print("  - 训练时请使用新的配置文件路径")
    else:
        print("⚠️  部分配置文件测试失败，请检查配置")

if __name__ == "__main__":
    main() 