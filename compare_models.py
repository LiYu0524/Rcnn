#!/usr/bin/env python3
"""
模型对比可视化脚本
同时显示Mask R-CNN和Sparse R-CNN的检测结果
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

class ModelComparator:
    def __init__(self, mask_rcnn_config, mask_rcnn_checkpoint, 
                 sparse_rcnn_config, sparse_rcnn_checkpoint):
        """初始化模型对比器"""
        print("加载Mask R-CNN模型...")
        self.mask_rcnn_model = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
        print("Mask R-CNN模型加载完成!")
        
        print("加载Sparse R-CNN模型...")
        self.sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')
        print("Sparse R-CNN模型加载完成!")
    
    def decode_mask(self, mask_pred, bbox, img_shape):
        """解码分割掩码"""
        try:
            if isinstance(mask_pred, dict):
                # RLE格式
                import pycocotools.mask as mask_util
                mask = mask_util.decode(mask_pred)
            else:
                # 数组格式
                mask = mask_pred
            
            if mask.shape != img_shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (img_shape[1], img_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return mask.astype(bool)
        except:
            # 如果解码失败，创建一个基于bbox的简单掩码
            mask = np.zeros(img_shape[:2], dtype=bool)
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_shape[1], x2), min(img_shape[0], y2)
            mask[y1:y2, x1:x2] = True
            return mask
    
    def visualize_result(self, img, result, ax, title, conf_threshold=0.3, show_masks=True):
        """可视化单个模型的结果"""
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        detection_count = 0
        
        if hasattr(result, 'pred_instances'):
            # 新版本MMDetection格式
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            # 获取分割掩码
            masks = None
            if hasattr(pred_instances, 'masks') and show_masks:
                masks = pred_instances.masks.cpu().numpy()
            
            # 创建用于掩码可视化的图像副本
            mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            
            # 绘制检测框和掩码
            for i in range(len(bboxes)):
                if scores[i] > conf_threshold:
                    x1, y1, x2, y2 = bboxes[i]
                    score = scores[i]
                    class_id = labels[i]
                    detection_count += 1
                    
                    # 确保class_id在有效范围内
                    if class_id < len(VOC_CLASSES):
                        color = COLORS[class_id]
                        color_255 = (np.array(color[:3]) * 255).astype(int)
                        
                        # 绘制分割掩码
                        if masks is not None and i < len(masks):
                            mask = self.decode_mask(masks[i], bboxes[i], img.shape)
                            
                            # 创建彩色掩码
                            colored_mask = np.zeros_like(mask_img)
                            colored_mask[mask] = color_255
                            
                            # 半透明覆盖
                            alpha = 0.4
                            mask_img = np.where(mask[..., None], 
                                              (1-alpha) * mask_img + alpha * colored_mask,
                                              mask_img)
                        
                        # 绘制边界框
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor=color,
                                               facecolor='none')
                        ax.add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        ax.text(x1, y1-5, label,
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor=color, alpha=0.7),
                               fontsize=10, color='black')
            
            # 如果有掩码，更新显示
            if masks is not None and show_masks:
                ax.imshow(mask_img.astype(np.uint8))
                
        else:
            # 旧版本格式处理
            if isinstance(result, tuple) and len(result) >= 2:
                bbox_result, segm_result = result[0], result[1] if show_masks else None
            else:
                bbox_result, segm_result = result, None
            
            # 创建用于掩码可视化的图像副本
            mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            
            # 绘制检测框和掩码
            for class_id, bboxes in enumerate(bbox_result):
                color = COLORS[class_id]
                color_255 = (np.array(color[:3]) * 255).astype(int)
                
                masks = segm_result[class_id] if segm_result else None
                
                for i, bbox in enumerate(bboxes):
                    if len(bbox) >= 5 and bbox[4] > conf_threshold:
                        x1, y1, x2, y2, score = bbox[:5]
                        detection_count += 1
                        
                        # 绘制分割掩码
                        if masks is not None and i < len(masks):
                            mask = self.decode_mask(masks[i], bbox[:4], img.shape)
                            
                            # 创建彩色掩码
                            colored_mask = np.zeros_like(mask_img)
                            colored_mask[mask] = color_255
                            
                            # 半透明覆盖
                            alpha = 0.4
                            mask_img = np.where(mask[..., None], 
                                              (1-alpha) * mask_img + alpha * colored_mask,
                                              mask_img)
                        
                        # 绘制边界框
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor=color,
                                               facecolor='none')
                        ax.add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        ax.text(x1, y1-5, label,
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor=color, alpha=0.7),
                               fontsize=10, color='black')
            
            # 如果有掩码，更新显示
            if segm_result is not None and show_masks:
                ax.imshow(mask_img.astype(np.uint8))
        
        return detection_count
    
    def compare_models(self, img_path, save_path=None, conf_threshold=0.3):
        """对比两个模型的检测结果"""
        # 读取图像
        img = mmcv.imread(img_path)
        print(f"处理图像: {img_path}")
        print(f"图像尺寸: {img.shape}")
        
        # 进行推理
        print("Mask R-CNN推理中...")
        mask_rcnn_result = inference_detector(self.mask_rcnn_model, img)
        print("Sparse R-CNN推理中...")
        sparse_rcnn_result = inference_detector(self.sparse_rcnn_model, img)
        print("推理完成!")
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        # 2. Mask R-CNN结果
        mask_rcnn_count = self.visualize_result(
            img, mask_rcnn_result, axes[1], 
            f'Mask R-CNN (Threshold > {conf_threshold})', 
            conf_threshold, show_masks=True
        )
        
        # 3. Sparse R-CNN结果
        sparse_rcnn_count = self.visualize_result(
            img, sparse_rcnn_result, axes[2], 
            f'Sparse R-CNN (Threshold > {conf_threshold})', 
            conf_threshold, show_masks=False
        )
        
        print(f"Mask R-CNN检测到 {mask_rcnn_count} 个目标")
        print(f"Sparse R-CNN检测到 {sparse_rcnn_count} 个目标")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比结果保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return mask_rcnn_count, sparse_rcnn_count
    
    def compare_multiple_images(self, img_paths, save_dir, conf_threshold=0.3):
        """对比多张图像的检测结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        total_mask_rcnn = 0
        total_sparse_rcnn = 0
        
        for i, img_path in enumerate(img_paths):
            save_path = save_dir / f"model_comparison_{i+1}.png"
            mask_count, sparse_count = self.compare_models(img_path, save_path, conf_threshold)
            total_mask_rcnn += mask_count
            total_sparse_rcnn += sparse_count
            print(f"图像 {i+1}/{len(img_paths)} 完成\n")
        
        print(f"总计 - Mask R-CNN: {total_mask_rcnn} 个目标")
        print(f"总计 - Sparse R-CNN: {total_sparse_rcnn} 个目标")

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
    parser = argparse.ArgumentParser(description='Mask R-CNN vs Sparse R-CNN模型对比')
    parser.add_argument('--mask-rcnn-config', 
                       default='work_dirs/mask_rcnn_pretrained/mask_rcnn_voc_pretrained_simple.py',
                       help='Mask R-CNN配置文件路径')
    parser.add_argument('--mask-rcnn-checkpoint', 
                       default='work_dirs/mask_rcnn_pretrained/best_coco_bbox_mAP_epoch_27.pth',
                       help='Mask R-CNN检查点文件路径')
    parser.add_argument('--sparse-rcnn-config', 
                       default='work_dirs/sparse_rcnn/sparse_rcnn_config.py',
                       help='Sparse R-CNN配置文件路径')
    parser.add_argument('--sparse-rcnn-checkpoint', 
                       default='work_dirs/sparse_rcnn/best_pascal_voc_mAP_epoch_40.pth',
                       help='Sparse R-CNN检查点文件路径')
    parser.add_argument('--img-dir', 
                       default='data/VOCdevkit/VOC2007/JPEGImages',
                       help='测试图像目录')
    parser.add_argument('--output-dir', 
                       default='visualizations/model_comparison',
                       help='输出目录')
    parser.add_argument('--num-images', type=int, default=4,
                       help='要对比的图像数量')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--single-img', type=str, default=None,
                       help='单张图像路径（可选）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.mask_rcnn_checkpoint):
        print(f"Mask R-CNN检查点文件不存在: {args.mask_rcnn_checkpoint}")
        return
    
    if not os.path.exists(args.sparse_rcnn_checkpoint):
        print(f"Sparse R-CNN检查点文件不存在: {args.sparse_rcnn_checkpoint}")
        return
    
    # 创建对比器
    try:
        comparator = ModelComparator(
            args.mask_rcnn_config, args.mask_rcnn_checkpoint,
            args.sparse_rcnn_config, args.sparse_rcnn_checkpoint
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.single_img:
        # 对比单张图像
        if os.path.exists(args.single_img):
            save_path = output_dir / "single_image_comparison.png"
            comparator.compare_models(args.single_img, save_path, args.conf_threshold)
        else:
            print(f"图像文件不存在: {args.single_img}")
    else:
        # 选择并对比多张图像
        test_images = select_test_images(args.img_dir, args.num_images)
        if test_images:
            print(f"选择了 {len(test_images)} 张图像进行对比")
            comparator.compare_multiple_images(test_images, output_dir, args.conf_threshold)
        else:
            print("没有找到测试图像")
    
    print("模型对比完成!")

if __name__ == "__main__":
    main() 