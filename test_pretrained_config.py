#!/usr/bin/env python3
"""
测试预训练Mask R-CNN配置文件是否正确
"""

import os
import sys
import traceback
from pathlib import Path

def test_config():
    """测试配置文件加载"""
    print("🧪 测试预训练Mask R-CNN配置文件...")
    
    config_file = "configs/mask_rcnn_voc_pretrained.py"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        # 添加mmdetection路径
        mmdet_path = os.path.abspath("mmdetection_repo")
        if mmdet_path not in sys.path:
            sys.path.insert(0, mmdet_path)
        
        # 导入MMDetection配置加载器
        from mmengine.config import Config
        
        # 加载配置
        print(f"📋 加载配置文件: {config_file}")
        cfg = Config.fromfile(config_file)
        
        print("✅ 配置文件加载成功!")
        
        # 检查关键配置
        print("\n📊 配置检查:")
        
        # 检查模型配置
        if hasattr(cfg, 'model'):
            print("✅ 模型配置存在")
            
            # 检查预训练权重
            if hasattr(cfg.model, 'init_cfg'):
                checkpoint = cfg.model.init_cfg.get('checkpoint', '')
                print(f"📦 预训练权重: {checkpoint}")
                
                # 检查权重文件是否存在
                if os.path.exists(checkpoint):
                    print("✅ 预训练权重文件存在")
                else:
                    print(f"⚠️  预训练权重文件不存在: {checkpoint}")
            
            # 检查ROI头配置
            if hasattr(cfg.model, 'roi_head'):
                roi_head = cfg.model.roi_head
                if hasattr(roi_head, 'bbox_head'):
                    bbox_classes = roi_head.bbox_head.get('num_classes', 0)
                    print(f"🎯 边界框类别数: {bbox_classes}")
                
                if hasattr(roi_head, 'mask_head'):
                    mask_classes = roi_head.mask_head.get('num_classes', 0)
                    print(f"🎭 掩码类别数: {mask_classes}")
        
        # 检查数据集配置
        if hasattr(cfg, 'train_dataloader'):
            dataset = cfg.train_dataloader.dataset
            ann_file = dataset.get('ann_file', '')
            print(f"📚 训练集标注: {ann_file}")
        
        if hasattr(cfg, 'val_dataloader'):
            dataset = cfg.val_dataloader.dataset
            ann_file = dataset.get('ann_file', '')
            print(f"📚 验证集标注: {ann_file}")
        
        # 检查优化器配置
        if hasattr(cfg, 'optim_wrapper'):
            optimizer = cfg.optim_wrapper.optimizer
            lr = optimizer.get('lr', 0)
            print(f"📈 学习率: {lr}")
            
            # 检查分层学习率
            if hasattr(cfg.optim_wrapper, 'paramwise_cfg'):
                print("✅ 分层学习率配置存在")
                custom_keys = cfg.optim_wrapper.paramwise_cfg.get('custom_keys', {})
                for key, config in custom_keys.items():
                    lr_mult = config.get('lr_mult', 1.0)
                    print(f"   {key}: {lr_mult}x")
        
        # 检查训练配置
        if hasattr(cfg, 'train_cfg'):
            max_epochs = cfg.train_cfg.get('max_epochs', 0)
            val_interval = cfg.train_cfg.get('val_interval', 0)
            print(f"🔄 最大轮次: {max_epochs}")
            print(f"🔄 验证间隔: {val_interval}")
        
        # 检查自定义钩子
        if hasattr(cfg, 'custom_hooks'):
            print(f"🪝 自定义钩子数量: {len(cfg.custom_hooks)}")
            for hook in cfg.custom_hooks:
                hook_type = hook.get('type', 'Unknown')
                print(f"   - {hook_type}")
        
        # 检查可视化配置
        if hasattr(cfg, 'vis_backends'):
            print(f"📊 可视化后端数量: {len(cfg.vis_backends)}")
            for backend in cfg.vis_backends:
                backend_type = backend.get('type', 'Unknown')
                if backend_type == 'TensorboardVisBackend':
                    save_dir = backend.get('save_dir', '')
                    print(f"   - Tensorboard: {save_dir}")
        
        print("\n✅ 配置文件测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        print("\n📋 详细错误信息:")
        traceback.print_exc()
        return False

def check_dependencies():
    """检查依赖项"""
    print("\n🔍 检查依赖项...")
    
    # 检查MMDetection
    mmdet_path = "mmdetection_repo"
    if os.path.exists(mmdet_path):
        print(f"✅ MMDetection路径存在: {mmdet_path}")
    else:
        print(f"❌ MMDetection路径不存在: {mmdet_path}")
        return False
    
    # 检查自定义钩子
    custom_hooks_file = "custom_hooks.py"
    if os.path.exists(custom_hooks_file):
        print(f"✅ 自定义钩子文件存在: {custom_hooks_file}")
    else:
        print(f"❌ 自定义钩子文件不存在: {custom_hooks_file}")
        return False
    
    # 检查数据目录
    data_dir = "data/VOCdevkit/VOC2012"
    if os.path.exists(data_dir):
        print(f"✅ VOC2012数据目录存在: {data_dir}")
        
        # 检查关键文件
        train_file = os.path.join(data_dir, "ImageSets/Main/train.txt")
        val_file = os.path.join(data_dir, "ImageSets/Main/val.txt")
        
        if os.path.exists(train_file):
            print(f"✅ 训练集文件存在: {train_file}")
        else:
            print(f"❌ 训练集文件不存在: {train_file}")
        
        if os.path.exists(val_file):
            print(f"✅ 验证集文件存在: {val_file}")
        else:
            print(f"❌ 验证集文件不存在: {val_file}")
    else:
        print(f"❌ VOC2012数据目录不存在: {data_dir}")
    
    return True

def main():
    print("🎯 预训练Mask R-CNN配置测试")
    print("="*50)
    
    # 检查依赖项
    deps_ok = check_dependencies()
    
    if deps_ok:
        # 测试配置文件
        config_ok = test_config()
        
        if config_ok:
            print("\n🎉 所有测试通过!")
            print("\n💡 下一步:")
            print("1. 运行: python train_pretrained_mask_rcnn.py --check-weights")
            print("2. 运行: python train_pretrained_mask_rcnn.py --train")
        else:
            print("\n❌ 配置测试失败，请检查配置文件")
    else:
        print("\n❌ 依赖项检查失败，请检查环境设置")

if __name__ == "__main__":
    main() 