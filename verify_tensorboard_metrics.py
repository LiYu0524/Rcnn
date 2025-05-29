#!/usr/bin/env python3
"""
验证Tensorboard配置是否会记录关键指标的脚本
"""

import os
import sys
from mmengine.config import Config

def verify_tensorboard_metrics():
    """验证配置文件是否会记录关键的Tensorboard指标"""
    
    print("🔍 验证Tensorboard指标记录配置")
    print("="*60)
    
    config_files = [
        ('Sparse R-CNN', 'sparse_rcnn_config_simple.py'),
        ('Mask R-CNN', 'configs/mask_rcnn_voc_simple.py')
    ]
    
    for model_name, config_file in config_files:
        print(f"\n📋 检查 {model_name} 配置: {config_file}")
        print("-" * 50)
        
        if not os.path.exists(config_file):
            print(f"❌ 配置文件不存在: {config_file}")
            continue
        
        try:
            # 加载配置文件
            cfg = Config.fromfile(config_file)
            
            # 1. 检查Tensorboard后端配置
            print("1️⃣ Tensorboard后端配置:")
            if hasattr(cfg, 'vis_backends'):
                tensorboard_backends = [b for b in cfg.vis_backends if b.get('type') == 'TensorboardVisBackend']
                if tensorboard_backends:
                    print("  ✅ 已配置TensorboardVisBackend")
                    for backend in tensorboard_backends:
                        save_dir = backend.get('save_dir', '默认目录')
                        print(f"    📁 保存目录: {save_dir}")
                else:
                    print("  ❌ 未找到TensorboardVisBackend配置")
            else:
                print("  ❌ 未找到vis_backends配置")
            
            # 2. 检查训练配置
            print("\n2️⃣ 训练Loss记录:")
            if hasattr(cfg, 'train_cfg'):
                print("  ✅ 训练配置存在")
                print("  📊 将记录以下训练Loss:")
                
                # 根据模型类型显示预期的loss
                model_type = cfg.model.get('type', 'Unknown')
                if model_type == 'SparseRCNN':
                    expected_losses = [
                        'loss_cls (分类损失)',
                        'loss_bbox (边界框回归损失)', 
                        'loss_iou (IoU损失)',
                        'loss (总损失)'
                    ]
                elif model_type == 'MaskRCNN':
                    expected_losses = [
                        'loss_cls (分类损失)',
                        'loss_bbox (边界框回归损失)',
                        'loss_mask (掩码损失)',
                        'loss_rpn_cls (RPN分类损失)',
                        'loss_rpn_bbox (RPN回归损失)',
                        'loss (总损失)'
                    ]
                else:
                    expected_losses = ['loss (总损失)']
                
                for loss in expected_losses:
                    print(f"    - {loss}")
            else:
                print("  ❌ 训练配置缺失")
            
            # 3. 检查验证配置
            print("\n3️⃣ 验证Loss和mAP记录:")
            if hasattr(cfg, 'val_cfg') and hasattr(cfg, 'train_cfg'):
                val_interval = cfg.train_cfg.get('val_interval', 0)
                if val_interval > 0:
                    print(f"  ✅ 验证配置存在 (每{val_interval}个epoch验证)")
                    print("  📊 将记录以下验证指标:")
                    
                    # 验证Loss
                    print("    🔸 验证Loss:")
                    for loss in expected_losses:
                        val_loss = loss.replace('loss', 'val_loss')
                        print(f"      - {val_loss}")
                    
                    # 验证mAP
                    print("    🔸 验证mAP:")
                    if hasattr(cfg, 'val_evaluator'):
                        evaluator = cfg.val_evaluator
                        if evaluator.get('type') == 'VOCMetric':
                            print("      - val_mAP (平均精度均值)")
                            print("      - val_mAP_50 (IoU=0.5时的mAP)")
                            print("      - val_mAP_75 (IoU=0.75时的mAP)")
                            print("      - val_AP_per_class (每个类别的AP)")
                        else:
                            print(f"      - 使用评估器: {evaluator.get('type')}")
                    else:
                        print("      ❌ 未找到验证评估器配置")
                else:
                    print("  ❌ 验证间隔为0，不会进行验证")
            else:
                print("  ❌ 验证配置缺失")
            
            # 4. 检查学习率记录
            print("\n4️⃣ 学习率记录:")
            if hasattr(cfg, 'param_scheduler'):
                print("  ✅ 学习率调度器配置存在")
                print("  📊 将记录学习率变化曲线")
            else:
                print("  ❌ 学习率调度器配置缺失")
            
            # 5. 检查日志处理器
            print("\n5️⃣ 日志处理器:")
            if hasattr(cfg, 'log_processor'):
                log_processor = cfg.log_processor
                window_size = log_processor.get('window_size', 50)
                by_epoch = log_processor.get('by_epoch', True)
                print(f"  ✅ 日志处理器配置存在")
                print(f"    📊 窗口大小: {window_size}")
                print(f"    📅 按epoch记录: {by_epoch}")
            else:
                print("  ❌ 日志处理器配置缺失")
                
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("📊 总结 - 预期的Tensorboard可视化内容:")
    print("="*60)
    print("🔸 训练指标 (Train/):")
    print("  - loss: 总训练损失")
    print("  - loss_cls: 分类损失")
    print("  - loss_bbox: 边界框回归损失")
    print("  - loss_iou: IoU损失 (Sparse R-CNN)")
    print("  - loss_mask: 掩码损失 (Mask R-CNN)")
    print("  - loss_rpn_*: RPN相关损失 (Mask R-CNN)")
    
    print("\n🔸 验证指标 (Val/):")
    print("  - val_loss: 总验证损失")
    print("  - val_loss_*: 各种验证损失")
    print("  - val_mAP: 验证集平均精度均值")
    print("  - val_mAP_50: IoU=0.5时的mAP")
    print("  - val_mAP_75: IoU=0.75时的mAP")
    
    print("\n🔸 其他指标:")
    print("  - lr: 学习率变化")
    print("  - time: 训练和验证时间")
    
    print("\n💡 使用方法:")
    print("1. 开始训练后，运行: python view_latest_tensorboard.py")
    print("2. 在浏览器中访问: http://localhost:6006")
    print("3. 查看SCALARS标签页中的各种指标曲线")

def main():
    print("=== Tensorboard 指标验证工具 ===")
    verify_tensorboard_metrics()

if __name__ == "__main__":
    main() 