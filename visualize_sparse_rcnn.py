#!/usr/bin/env python3
"""
简化的Sparse R-CNN可视化脚本
专门用于查看Sparse R-CNN的目标检测结果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from pathlib import Path
import argparse
import random

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 为每个类别定义颜色
COLORS = plt.cm.Set3(np.linspace(0, 1, len(VOC_CLASSES)))

class SparseRCNNVisualizer:
    def __init__(self, config_path, checkpoint_path):
        """初始化Sparse R-CNN可视化器"""
        print(f"加载Sparse R-CNN模型...")
        print(f"配置文件: {config_path}")
        print(f"检查点文件: {checkpoint_path}")
        
        self.model = init_detector(config_path, checkpoint_path, device='cuda:0')
        print("模型加载完成!")
        
    def visualize_detection(self, img_path, save_path=None, conf_threshold=0.3):
        """可视化单张图像的检测结果"""
        # 读取图像
        img = mmcv.imread(img_path)
        print(f"处理图像: {img_path}")
        print(f"图像尺寸: {img.shape}")
        
        # 进行推理
        print("开始推理...")
        result = inference_detector(self.model, img)
        print("推理完成!")
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        # 检测结果
        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Sparse R-CNN (Threshold > {conf_threshold})', fontsize=14)
        axes[1].axis('off')
        
        # 解析检测结果 - 适配新版本MMDetection
        detection_count = 0
        
        # 检查结果类型并提取预测数据
        if hasattr(result, 'pred_instances'):
            # 新版本MMDetection格式
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            print(f"检测到 {len(bboxes)} 个候选目标")
            
            # 绘制检测框
            for i in range(len(bboxes)):
                if scores[i] > conf_threshold:
                    x1, y1, x2, y2 = bboxes[i]
                    score = scores[i]
                    class_id = labels[i]
                    detection_count += 1
                    
                    # 确保class_id在有效范围内
                    if class_id < len(VOC_CLASSES):
                        color = COLORS[class_id]
                        
                        # 绘制边界框
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor=color,
                                               facecolor='none')
                        axes[1].add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        axes[1].text(x1, y1-5, label,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=color, alpha=0.7),
                                   fontsize=10, color='black')
        else:
            # 旧版本格式处理
            if isinstance(result, tuple):
                bbox_result = result[0]
            else:
                bbox_result = result
                
            # 绘制检测框
            for class_id, bboxes in enumerate(bbox_result):
                color = COLORS[class_id]
                for bbox in bboxes:
                    if len(bbox) >= 5 and bbox[4] > conf_threshold:
                        x1, y1, x2, y2, score = bbox[:5]
                        detection_count += 1
                        
                        # 绘制边界框
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor=color,
                                               facecolor='none')
                        axes[1].add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        axes[1].text(x1, y1-5, label,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=color, alpha=0.7),
                                   fontsize=10, color='black')
        
        print(f"检测到 {detection_count} 个目标")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return detection_count
    
    def visualize_multiple_images(self, img_paths, save_dir, conf_threshold=0.3):
        """可视化多张图像的检测结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        total_detections = 0
        for i, img_path in enumerate(img_paths):
            save_path = save_dir / f"sparse_rcnn_result_{i+1}.png"
            detections = self.visualize_detection(img_path, save_path, conf_threshold)
            total_detections += detections
            print(f"图像 {i+1}/{len(img_paths)} 完成\n")
        
        print(f"总共检测到 {total_detections} 个目标")

def select_test_images(voc_test_dir, num_images=4):
    """从VOC测试集中选择图像"""
    test_dir = Path(voc_test_dir)
    if not test_dir.exists():
        print(f"测试目录不存在: {test_dir}")
        return []
    
    # 获取所有jpg图像
    jpg_files = list(test_dir.glob("*.jpg"))
    if len(jpg_files) < num_images:
        print(f"测试目录中图像数量不足，需要{num_images}张，只找到{len(jpg_files)}张")
        return [str(path) for path in jpg_files]
    
    # 随机选择指定数量的图像
    selected = random.sample(jpg_files, num_images)
    return [str(path) for path in selected]

def main():
    parser = argparse.ArgumentParser(description='Sparse R-CNN检测结果可视化')
    parser.add_argument('--config', 
                       default='work_dirs/sparse_rcnn/sparse_rcnn_config.py',
                       help='Sparse R-CNN配置文件路径')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/sparse_rcnn/best_pascal_voc_mAP_epoch_40.pth',
                       help='Sparse R-CNN检查点文件路径')
    parser.add_argument('--img-dir', 
                       default='data/VOCdevkit/VOC2007/JPEGImages',
                       help='测试图像目录')
    parser.add_argument('--output-dir', 
                       default='visualizations/sparse_rcnn_results',
                       help='输出目录')
    parser.add_argument('--num-images', type=int, default=4,
                       help='要可视化的图像数量')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--single-img', type=str, default=None,
                       help='单张图像路径（可选）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"检查点文件不存在: {args.checkpoint}")
        print("可用的检查点文件:")
        checkpoint_dir = Path(args.checkpoint).parent
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  {f}")
        return
    
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return
    
    # 创建可视化器
    try:
        visualizer = SparseRCNNVisualizer(args.config, args.checkpoint)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.single_img:
        # 可视化单张图像
        if os.path.exists(args.single_img):
            save_path = output_dir / "single_image_result.png"
            visualizer.visualize_detection(args.single_img, save_path, args.conf_threshold)
        else:
            print(f"图像文件不存在: {args.single_img}")
    else:
        # 选择并可视化多张图像
        test_images = select_test_images(args.img_dir, args.num_images)
        if test_images:
            print(f"选择了 {len(test_images)} 张图像进行可视化")
            visualizer.visualize_multiple_images(test_images, output_dir, args.conf_threshold)
        else:
            print("没有找到测试图像")
    
    print("可视化完成!")

if __name__ == "__main__":
    main() 