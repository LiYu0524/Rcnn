#!/usr/bin/env python3
"""
Mask R-CNN可视化脚本
功能：
1. 显示经过NMS后的高质量RPN proposals
2. 显示最终检测结果（边界框+类别+置信度）
3. 显示实例分割掩码（半透明覆盖）
布局：三张图并排（原图 | Proposals | 最终结果+分割）
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
import random
from mmcv.ops import nms
import pycocotools.mask as mask_util

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 为每个类别定义颜色
COLORS = plt.cm.Set3(np.linspace(0, 1, len(VOC_CLASSES)))

class MaskRCNNVisualizer:
    def __init__(self, config_path, checkpoint_path):
        """初始化Mask R-CNN可视化器"""
        print(f"加载Mask R-CNN模型...")
        print(f"配置文件: {config_path}")
        print(f"检查点文件: {checkpoint_path}")
        
        self.model = init_detector(config_path, checkpoint_path, device='cuda:0')
        print("模型加载完成!")
        
        # 用于存储RPN proposals的变量
        self.rpn_proposals = None
        self.register_hooks()
        
    def register_hooks(self):
        """注册hook来捕获RPN的输出"""
        def rpn_hook(module, input, output):
            """捕获RPN的proposals"""
            print(f"Hook触发! 模块: {type(module).__name__}")
            print(f"输出类型: {type(output)}")
            
            # 只处理RPNHead的输出
            if type(module).__name__ == 'RPNHead':
                print("检测到RPNHead输出!")
                if isinstance(output, tuple) and len(output) >= 2:
                    print(f"RPNHead输出元组长度: {len(output)}")
                    
                    # RPNHead在推理时的输出格式通常是 (cls_scores, bbox_preds)
                    # 但我们需要的是处理后的proposals
                    
                    # 尝试访问模块的其他属性来获取proposals
                    if hasattr(module, '_get_proposals_single'):
                        print("尝试通过_get_proposals_single获取proposals...")
                        
                    # 检查输入参数，有时proposals在input中
                    if input and len(input) > 0:
                        print(f"检查输入参数数量: {len(input)}")
                        for i, inp in enumerate(input):
                            if hasattr(inp, 'shape'):
                                print(f"  输入[{i}]: {type(inp)} - {inp.shape}")
                            else:
                                print(f"  输入[{i}]: {type(inp)}")
                    
                    # 检查输出
                    for i, item in enumerate(output):
                        if isinstance(item, list):
                            print(f"  输出[{i}]: list with {len(item)} items")
                            if len(item) > 0:
                                first_item = item[0]
                                if hasattr(first_item, 'shape'):
                                    print(f"    第一项形状: {first_item.shape}")
                                    print(f"    第一项类型: {type(first_item)}")
                        elif hasattr(item, 'shape'):
                            print(f"  输出[{i}]: {type(item)} - {item.shape}")
        
        def prediction_hook(module, input, output):
            """捕获预测结果中的proposals"""
            if 'predict' in type(module).__name__.lower() or 'proposal' in type(module).__name__.lower():
                print(f"预测Hook触发! 模块: {type(module).__name__}")
                
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    for i, item in enumerate(output):
                        if hasattr(item, 'bboxes'):
                            proposals = item.bboxes.cpu().numpy()
                            print(f"在预测输出[{i}]找到proposals: {proposals.shape}")
                            if proposals.shape[-1] >= 4:  # 确保是有效的bbox格式
                                self.rpn_proposals = proposals
                                print(f"成功捕获 {len(proposals)} 个proposals")
                                return
        
        # 清理之前的hook
        print("模型结构中的模块:")
        rpn_hook_registered = False
        
        for name, module in self.model.named_modules():
            # 只显示关键模块，避免输出过多
            if any(x in name.lower() for x in ['rpn', 'head', 'neck']):
                print(f"  {name}: {type(module).__name__}")
            
            # 专门针对RPNHead注册hook
            if type(module).__name__ == 'RPNHead':
                print(f"在 RPNHead 注册专用hook")
                module.register_forward_hook(rpn_hook)
                rpn_hook_registered = True
            
            # 也尝试在proposal相关的方法上注册hook
            if 'predict' in name.lower() and hasattr(module, 'forward'):
                print(f"在 {name} 注册预测hook")
                module.register_forward_hook(prediction_hook)
        
        if not rpn_hook_registered:
            print("警告: 没有找到RPNHead模块")
        else:
            print("RPNHead hook注册成功")
    
    def get_rpn_proposals_alternative(self, img):
        """备用方法：通过模型的test_step获取proposals"""
        print("尝试备用方法获取proposals...")
        
        try:
            # 方法1: 尝试新版本MMDetection的方式
            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            
            data_sample = DetDataSample()
            data_sample.set_metainfo({'img_shape': img.shape[:2], 'ori_shape': img.shape[:2]})
            
            # 使用模型的data_preprocessor
            data = {'inputs': [torch.from_numpy(img).permute(2, 0, 1).float().cuda()], 
                   'data_samples': [data_sample]}
            data = self.model.data_preprocessor(data)
            
            # 提取特征
            x = self.model.extract_feat(data['inputs'])
            print(f"特征提取完成，特征层数: {len(x) if isinstance(x, (list, tuple)) else 1}")
            
            # 尝试获取RPN proposals
            if hasattr(self.model, 'rpn_head'):
                print("找到rpn_head，尝试获取proposals...")
                rpn_results_list = self.model.rpn_head.predict(x, data['data_samples'])
                
                if rpn_results_list and len(rpn_results_list) > 0:
                    proposals = rpn_results_list[0].bboxes.cpu().numpy()
                    scores = rpn_results_list[0].scores.cpu().numpy()
                    
                    # 按分数排序，取top proposals
                    sorted_indices = np.argsort(scores)[::-1]
                    top_proposals = proposals[sorted_indices[:200]]  # 取前200个
                    
                    print(f"方法1成功: 获取到 {len(top_proposals)} 个proposals")
                    return top_proposals
            
        except Exception as e:
            print(f"方法1失败: {e}")
        
        try:
            # 方法2: 尝试更直接的方式
            print("尝试方法2...")
            
            # 预处理图像
            from mmcv.transforms import Compose
            from mmdet.datasets.transforms import LoadImageFromFile, Resize, Normalize, PackDetInputs
            
            # 简化的预处理
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
            
            # 提取特征
            with torch.no_grad():
                features = self.model.extract_feat(img_tensor)
                print(f"特征提取成功: {len(features) if isinstance(features, (list, tuple)) else 1} 个特征层")
                
                # 尝试从不同的头部获取proposals
                if hasattr(self.model, 'rpn_head'):
                    # 创建假的data_samples
                    data_samples = [type('obj', (object,), {
                        'metainfo': {'img_shape': img.shape[:2], 'ori_shape': img.shape[:2]},
                        'img_shape': img.shape[:2],
                        'ori_shape': img.shape[:2]
                    })()]
                    
                    try:
                        rpn_outputs = self.model.rpn_head(features)
                        print(f"RPN输出类型: {type(rpn_outputs)}")
                        
                        if isinstance(rpn_outputs, tuple) and len(rpn_outputs) >= 2:
                            cls_scores, bbox_preds = rpn_outputs[0], rpn_outputs[1]
                            print(f"分类分数形状: {[x.shape for x in cls_scores] if isinstance(cls_scores, list) else cls_scores.shape}")
                            print(f"回归预测形状: {[x.shape for x in bbox_preds] if isinstance(bbox_preds, list) else bbox_preds.shape}")
                            
                            # 尝试解码bbox
                            proposals = self.model.rpn_head.predict_by_feat(
                                cls_scores, bbox_preds, 
                                batch_img_metas=[{'img_shape': img.shape[:2], 'ori_shape': img.shape[:2]}]
                            )
                            
                            if proposals and len(proposals) > 0:
                                if hasattr(proposals[0], 'bboxes'):
                                    bboxes = proposals[0].bboxes.cpu().numpy()
                                    scores = proposals[0].scores.cpu().numpy()
                                    
                                    # 按分数排序
                                    sorted_indices = np.argsort(scores)[::-1]
                                    top_proposals = bboxes[sorted_indices[:200]]
                                    
                                    print(f"方法2成功: 获取到 {len(top_proposals)} 个proposals")
                                    return top_proposals
                    except Exception as e2:
                        print(f"RPN预测失败: {e2}")
                
        except Exception as e:
            print(f"方法2失败: {e}")
        
        try:
            # 方法3: 生成网格proposal作为fallback
            print("尝试方法3: 生成网格proposals...")
            h, w = img.shape[:2]
            proposals = []
            
            # 生成多尺度网格proposals
            scales = [32, 64, 128, 256]
            for scale in scales:
                step_x = step_y = scale // 2
                for y in range(0, h - scale, step_y):
                    for x in range(0, w - scale, step_x):
                        proposals.append([x, y, x + scale, y + scale])
            
            proposals = np.array(proposals, dtype=np.float32)
            
            # 随机选择一些proposals
            if len(proposals) > 200:
                indices = np.random.choice(len(proposals), 200, replace=False)
                proposals = proposals[indices]
            
            print(f"方法3: 生成了 {len(proposals)} 个网格proposals")
            return proposals
            
        except Exception as e:
            print(f"方法3失败: {e}")
            return None
    
    def decode_mask(self, mask_pred, bbox, img_shape):
        """解码分割掩码"""
        try:
            if isinstance(mask_pred, dict):
                # RLE格式
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
    
    def visualize_detection(self, img_path, save_path=None, conf_threshold=0.3, max_proposals=50):
        """可视化单张图像的检测结果"""
        # 读取图像
        img = mmcv.imread(img_path)
        print(f"处理图像: {img_path}")
        print(f"图像尺寸: {img.shape}")
        
        # 重置proposals
        self.rpn_proposals = None
        
        # 进行推理
        print("开始推理...")
        print("=" * 50)
        result = inference_detector(self.model, img)
        print("=" * 50)
        print("推理完成!")
        
        # 检查hook是否捕获到了proposals
        print(f"Hook捕获状态: {self.rpn_proposals is not None}")
        if self.rpn_proposals is not None:
            print(f"Hook捕获到的proposals数量: {len(self.rpn_proposals)}")
            print(f"Proposals形状: {self.rpn_proposals.shape}")
        
        # 如果hook没有捕获到proposals，使用备用方法
        if self.rpn_proposals is None:
            print("Hook未捕获到proposals，尝试备用方法...")
            self.rpn_proposals = self.get_rpn_proposals_alternative(img)
        
        # 创建三张图的可视化
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        # 2. RPN Proposals
        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'RPN Proposals (Top {max_proposals})', fontsize=14)
        axes[1].axis('off')
        
        # 绘制proposals
        if self.rpn_proposals is not None:
            print(f"准备显示proposals，总数: {len(self.rpn_proposals)}")
            print(f"Proposals形状: {self.rpn_proposals.shape}")
            
            # 检查proposals的形状是否正确
            if len(self.rpn_proposals.shape) != 2 or self.rpn_proposals.shape[1] < 4:
                print(f"警告: proposals形状不正确 {self.rpn_proposals.shape}，期望 (N, 4) 或 (N, 5)")
                axes[1].text(0.5, 0.5, f'Invalid Proposals\nShape: {self.rpn_proposals.shape}', 
                            transform=axes[1].transAxes, ha='center', va='center',
                            fontsize=16, color='red')
            else:
                # 限制显示的proposals数量
                proposals_to_show = self.rpn_proposals[:max_proposals]
                valid_proposals = 0
                
                for i, proposal in enumerate(proposals_to_show):
                    if len(proposal) >= 4:
                        x1, y1, x2, y2 = proposal[:4]
                        
                        # 将坐标转换为标量并验证
                        try:
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                            # 确保坐标有效
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                       linewidth=1, edgecolor='red',
                                                       facecolor='none', alpha=0.7)
                                axes[1].add_patch(rect)
                                valid_proposals += 1
                        except (ValueError, TypeError) as e:
                            print(f"跳过无效proposal {i}: {e}")
                            continue
                
                print(f"显示了 {valid_proposals} 个有效proposals (总共{len(proposals_to_show)}个)")
        else:
            axes[1].text(0.5, 0.5, 'No Proposals\nCaptured', 
                        transform=axes[1].transAxes, ha='center', va='center',
                        fontsize=16, color='red')
            print("没有proposals可显示")
        
        # 3. 最终检测结果 + 实例分割
        axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Final Results + Segmentation (Threshold > {conf_threshold})', fontsize=14)
        axes[2].axis('off')
        
        # 解析检测结果
        print(f"检测结果类型: {type(result)}")
        detection_count = 0
        
        if hasattr(result, 'pred_instances'):
            # 新版本MMDetection格式
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            # 获取分割掩码
            masks = None
            if hasattr(pred_instances, 'masks'):
                masks = pred_instances.masks.cpu().numpy()
            
            print(f"检测到 {len(bboxes)} 个候选目标")
            
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
                        axes[2].add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        axes[2].text(x1, y1-5, label,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=color, alpha=0.7),
                                   fontsize=10, color='black')
            
            # 更新第三张图的显示
            axes[2].imshow(mask_img.astype(np.uint8))
            
        else:
            # 旧版本格式处理
            print("使用旧版本结果格式")
            if isinstance(result, tuple) and len(result) >= 2:
                bbox_result, segm_result = result[0], result[1]
            else:
                bbox_result, segm_result = result, None
            
            # 创建用于掩码可视化的图像副本
            mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            
            # 绘制检测框和掩码
            for class_id, (bboxes, masks) in enumerate(zip(bbox_result, segm_result or [None]*len(bbox_result))):
                color = COLORS[class_id]
                color_255 = (np.array(color[:3]) * 255).astype(int)
                
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
                        axes[2].add_patch(rect)
                        
                        # 添加类别标签和分数
                        label = f'{VOC_CLASSES[class_id]}: {score:.2f}'
                        axes[2].text(x1, y1-5, label,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor=color, alpha=0.7),
                                   fontsize=10, color='black')
            
            # 更新第三张图的显示
            axes[2].imshow(mask_img.astype(np.uint8))
        
        print(f"最终检测到 {detection_count} 个目标")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return detection_count
    
    def visualize_multiple_images(self, img_paths, save_dir, conf_threshold=0.3, max_proposals=50):
        """可视化多张图像的检测结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        total_detections = 0
        for i, img_path in enumerate(img_paths):
            save_path = save_dir / f"mask_rcnn_result_{i+1}.png"
            detections = self.visualize_detection(img_path, save_path, conf_threshold, max_proposals)
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
    parser = argparse.ArgumentParser(description='Mask R-CNN检测和分割结果可视化')
    parser.add_argument('--config', 
                       default='./work_dirs/mask_rcnn_coco_voc/mask_rcnn_coco_voc.py',
                       help='Mask R-CNN配置文件路径')
    parser.add_argument('--checkpoint', 
                       default='./work_dirs/mask_rcnn_coco_voc/best_coco_segm_mAP_epoch_40.pth',
                       help='Mask R-CNN检查点文件路径')
    parser.add_argument('--img-dir', 
                       default='data/VOCdevkit/VOC2007/JPEGImages',
                       help='测试图像目录')
    parser.add_argument('--output-dir', 
                       default='visualizations/mask_rcnn_results',
                       help='输出目录')
    parser.add_argument('--num-images', type=int, default=4,
                       help='要可视化的图像数量')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--max-proposals', type=int, default=50,
                       help='显示的最大proposal数量')
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
        visualizer = MaskRCNNVisualizer(args.config, args.checkpoint)
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
            visualizer.visualize_detection(args.single_img, save_path, 
                                         args.conf_threshold, args.max_proposals)
        else:
            print(f"图像文件不存在: {args.single_img}")
    else:
        # 选择并可视化多张图像
        test_images = select_test_images(args.img_dir, args.num_images)
        if test_images:
            print(f"选择了 {len(test_images)} 张图像进行可视化")
            visualizer.visualize_multiple_images(test_images, output_dir, 
                                               args.conf_threshold, args.max_proposals)
        else:
            print("没有找到测试图像")
    
    print("可视化完成!")

if __name__ == "__main__":
    main() 