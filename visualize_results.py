#!/usr/bin/env python3
"""
可视化脚本：对比Mask R-CNN和Sparse R-CNN的检测结果
包括Mask R-CNN的proposal box可视化
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from pathlib import Path
import argparse

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class ModelVisualizer:
    def __init__(self, mask_rcnn_config, mask_rcnn_checkpoint, 
                 sparse_rcnn_config, sparse_rcnn_checkpoint):
        """初始化可视化器"""
        self.mask_rcnn_model = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')
        self.sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')
        
    def visualize_mask_rcnn_proposals(self, img_path, save_path):
        """可视化Mask R-CNN的proposal boxes和最终结果"""
        img = mmcv.imread(img_path)
        
        # 获取模型推理结果
        result = inference_detector(self.mask_rcnn_model, img)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 获取RPN proposals（需要修改模型以输出proposals）
        # 这里我们用一个简化的方法来模拟proposal可视化
        self._visualize_proposals(img, axes[1])
        
        # 最终检测结果
        self._visualize_detection_result(img, result, axes[2], "Mask R-CNN最终结果")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _visualize_proposals(self, img, ax):
        """可视化proposal boxes（简化版本）"""
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('RPN Proposals (模拟)')
        ax.axis('off')
        
        # 这里添加一些模拟的proposal boxes
        # 在实际实现中，需要修改模型来输出RPN proposals
        h, w = img.shape[:2]
        for i in range(10):  # 显示10个模拟proposals
            x = np.random.randint(0, w-100)
            y = np.random.randint(0, h-100)
            width = np.random.randint(50, 150)
            height = np.random.randint(50, 150)
            
            rect = patches.Rectangle((x, y), width, height, 
                                   linewidth=1, edgecolor='red', 
                                   facecolor='none', alpha=0.7)
            ax.add_patch(rect)
    
    def _visualize_detection_result(self, img, result, ax, title):
        """可视化检测结果"""
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
        
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
            
        # 可视化边界框
        for class_id, bboxes in enumerate(bbox_result):
            for bbox in bboxes:
                if len(bbox) == 5 and bbox[4] > 0.3:  # 置信度阈值
                    x1, y1, x2, y2, score = bbox
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor='green',
                                           facecolor='none')
                    ax.add_patch(rect)
                    
                    # 添加类别标签和分数
                    ax.text(x1, y1-5, f'{VOC_CLASSES[class_id]}: {score:.2f}',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        # 可视化分割掩码
        if segm_result is not None:
            for class_id, masks in enumerate(segm_result):
                for mask in masks:
                    if isinstance(mask, dict):
                        # RLE格式
                        mask = mask['counts']
                    # 这里需要解码RLE格式的掩码
                    # 简化处理，实际使用时需要proper RLE解码
                    pass
    
    def compare_models(self, img_path, save_path):
        """对比两个模型的检测结果"""
        img = mmcv.imread(img_path)
        
        # 获取两个模型的结果
        mask_rcnn_result = inference_detector(self.mask_rcnn_model, img)
        sparse_rcnn_result = inference_detector(self.sparse_rcnn_model, img)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # Mask R-CNN结果
        self._visualize_detection_result(img, mask_rcnn_result, axes[1], "Mask R-CNN")
        
        # Sparse R-CNN结果
        self._visualize_detection_result(img, sparse_rcnn_result, axes[2], "Sparse R-CNN")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def test_external_images(self, img_paths, save_dir):
        """在外部图像上测试模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(img_paths):
            save_path = save_dir / f"external_test_{i+1}.png"
            self.compare_models(img_path, save_path)
            print(f"外部图像 {i+1} 结果保存到: {save_path}")

def download_external_images():
    """下载包含VOC类别的外部测试图像"""
    import urllib.request
    
    # 一些包含VOC类别物体的图像URL（示例）
    image_urls = [
        "https://example.com/car_image.jpg",  # 包含汽车
        "https://example.com/person_dog.jpg",  # 包含人和狗
        "https://example.com/airplane.jpg"    # 包含飞机
    ]
    
    external_dir = Path("external_images")
    external_dir.mkdir(exist_ok=True)
    
    downloaded_paths = []
    for i, url in enumerate(image_urls):
        try:
            save_path = external_dir / f"external_{i+1}.jpg"
            urllib.request.urlretrieve(url, save_path)
            downloaded_paths.append(str(save_path))
            print(f"下载图像 {i+1}: {save_path}")
        except Exception as e:
            print(f"下载图像 {i+1} 失败: {e}")
    
    return downloaded_paths

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
        return jpg_files
    
    # 随机选择指定数量的图像
    import random
    selected = random.sample(jpg_files, num_images)
    return [str(path) for path in selected]

def main():
    parser = argparse.ArgumentParser(description='可视化模型检测结果')
    parser.add_argument('--mask-rcnn-config', default='configs/mask_rcnn_voc.py',
                       help='Mask R-CNN配置文件路径')
    parser.add_argument('--mask-rcnn-checkpoint', default='work_dirs/mask_rcnn/latest.pth',
                       help='Mask R-CNN检查点文件路径')
    parser.add_argument('--sparse-rcnn-config', default='sparse_rcnn_config.py',
                       help='Sparse R-CNN配置文件路径')
    parser.add_argument('--sparse-rcnn-checkpoint', default='work_dirs/sparse_rcnn/latest.pth',
                       help='Sparse R-CNN检查点文件路径')
    parser.add_argument('--voc-test-dir', default='data/VOCdevkit/VOC2007/JPEGImages',
                       help='VOC测试图像目录')
    parser.add_argument('--output-dir', default='visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.mask_rcnn_checkpoint):
        print(f"Mask R-CNN检查点文件不存在: {args.mask_rcnn_checkpoint}")
        return
    
    if not os.path.exists(args.sparse_rcnn_checkpoint):
        print(f"Sparse R-CNN检查点文件不存在: {args.sparse_rcnn_checkpoint}")
        return
    
    # 创建可视化器
    visualizer = ModelVisualizer(
        args.mask_rcnn_config, args.mask_rcnn_checkpoint,
        args.sparse_rcnn_config, args.sparse_rcnn_checkpoint
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 选择VOC测试集中的4张图像进行可视化
    test_images = select_test_images(args.voc_test_dir, 4)
    if test_images:
        print("开始可视化VOC测试集图像...")
        for i, img_path in enumerate(test_images):
            # Mask R-CNN proposal可视化
            proposal_save_path = output_dir / f"mask_rcnn_proposals_{i+1}.png"
            visualizer.visualize_mask_rcnn_proposals(img_path, proposal_save_path)
            print(f"Mask R-CNN proposals结果保存到: {proposal_save_path}")
            
            # 模型对比
            compare_save_path = output_dir / f"model_comparison_{i+1}.png"
            visualizer.compare_models(img_path, compare_save_path)
            print(f"模型对比结果保存到: {compare_save_path}")
    
    # 2. 下载并测试外部图像
    print("下载外部测试图像...")
    external_images = download_external_images()
    if external_images:
        visualizer.test_external_images(external_images, output_dir / "external_tests")
    
    print("可视化完成!")

if __name__ == "__main__":
    main() 