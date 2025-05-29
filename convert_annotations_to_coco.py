#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将final_annotation_generator.py生成的标注转换为MMDetection COCO格式
"""

import os
import json
import pickle
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import pycocotools.mask as mask_util


class AnnotationToCOCOConverter:
    """将pickle格式的标注转换为COCO格式"""
    
    def __init__(self, 
                 annotation_dir="./mask_rcnn_in_tf2_keras/final_annotations",
                 output_dir="./data/coco_format",
                 voc_data_path="/mnt/data/liyu/mm/data/VOCdevkit/VOC2012"):
        """
        初始化转换器
        
        Args:
            annotation_dir: pickle标注文件目录
            output_dir: COCO格式输出目录
            voc_data_path: VOC原始数据路径
        """
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.voc_data_path = voc_data_path
        
        # VOC类别（包含背景类）
        self.classes = [
            '_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # COCO类别（不包含背景类）
        self.coco_categories = []
        for i, class_name in enumerate(self.classes[1:], 1):  # 跳过背景类
            self.coco_categories.append({
                "id": i,
                "name": class_name,
                "supercategory": "object"
            })
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "annotations")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_dir, "images")).mkdir(exist_ok=True)
        
        print(f"标注转换器初始化完成")
        print(f"  输入目录: {annotation_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  VOC数据路径: {voc_data_path}")
        print(f"  类别数量: {len(self.coco_categories)}")
    
    def _load_pickle_data(self, split):
        """加载pickle格式的标注数据"""
        pickle_file = os.path.join(self.annotation_dir, f"annotations_{split}.pkl")
        
        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"找不到标注文件: {pickle_file}")
        
        print(f"加载 {split} 数据: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def _get_original_image_info(self, file_name):
        """获取原始图像信息"""
        # 从文件名中提取图像ID
        image_id = os.path.splitext(file_name)[0]
        
        # 构建原始图像路径
        original_image_path = os.path.join(self.voc_data_path, "JPEGImages", file_name)
        
        # 对于使用mini mask的数据，所有标注都是基于320x320的处理后尺寸
        # 因此图像信息也应该使用320x320，与调整后的图像文件保持一致
        return {
            "id": hash(image_id) % (2**31),  # 生成唯一ID
            "file_name": file_name,
            "width": 320,   # 使用处理后的固定尺寸
            "height": 320,  # 使用处理后的固定尺寸
            "original_path": original_image_path
        }
    
    def _decode_mini_mask(self, mini_mask, bbox, original_size=(320, 320)):
        """将mini mask解码为全尺寸mask"""
        ymin, xmin, ymax, xmax = bbox.astype(int)
        
        # 确保边界框有效
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(original_size[0], ymax)
        xmax = min(original_size[1], xmax)
        
        if ymax <= ymin or xmax <= xmin:
            return np.zeros(original_size, dtype=np.uint8)
        
        # 调整mini mask到边界框尺寸
        roi_height = ymax - ymin
        roi_width = xmax - xmin
        
        if mini_mask.size > 0:
            mask_resized = cv2.resize(mini_mask.astype(np.float32), 
                                    (roi_width, roi_height), 
                                    interpolation=cv2.INTER_LINEAR)
            mask_resized = (mask_resized >= 0.5).astype(np.uint8)
        else:
            mask_resized = np.zeros((roi_height, roi_width), dtype=np.uint8)
        
        # 创建全尺寸mask
        full_mask = np.zeros(original_size, dtype=np.uint8)
        full_mask[ymin:ymax, xmin:xmax] = mask_resized
        
        return full_mask
    
    def _mask_to_rle(self, mask):
        """将mask转换为RLE格式"""
        # 确保mask是二值的
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 转换为Fortran顺序（列优先）
        binary_mask = np.asfortranarray(binary_mask)
        
        # 编码为RLE
        rle = mask_util.encode(binary_mask)
        
        # 确保counts是字符串格式
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle
    
    def _copy_images_to_output(self, file_names, split):
        """复制并调整图像到输出目录"""
        print(f"复制并调整 {split} 图像到输出目录...")
        
        copied_count = 0
        for file_name in file_names:
            src_path = os.path.join(self.voc_data_path, "JPEGImages", file_name)
            dst_path = os.path.join(self.output_dir, "images", file_name)
            
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    # 读取原始图像
                    image = cv2.imread(src_path)
                    if image is None:
                        print(f"  警告: 无法读取图像 {file_name}")
                        continue
                    
                    # 调整图像尺寸为320x320（不保持比例，与标注数据一致）
                    resized_image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
                    
                    # 保存调整后的图像
                    cv2.imwrite(dst_path, resized_image)
                    copied_count += 1
                    
                except Exception as e:
                    print(f"  警告: 处理图像失败 {file_name}: {e}")
        
        print(f"  成功处理 {copied_count} 张图像（调整为320x320）")
    
    def convert_split(self, split):
        """转换指定分割的数据"""
        print(f"\n开始转换 {split} 数据...")
        
        # 加载pickle数据
        data = self._load_pickle_data(split)
        
        # 提取数据
        batch_images = data['images']
        batch_masks = data['masks']
        batch_gt_boxes = data['gt_boxes']
        batch_labels = data['labels']
        file_names = data['file_names']
        statistics = data['statistics']
        
        print(f"数据统计:")
        print(f"  批次数量: {len(batch_images)}")
        print(f"  图像数量: {len(file_names)}")
        print(f"  批次大小: {statistics['batch_size']}")
        print(f"  图像尺寸: {statistics['image_size']}")
        print(f"  使用mini mask: {statistics['use_mini_mask']}")
        
        # 初始化COCO格式数据
        coco_data = {
            "info": {
                "description": f"VOC2012 {split} dataset converted from pickle format",
                "version": "1.0",
                "year": 2024,
                "contributor": "Annotation Converter",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "categories": self.coco_categories,
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        image_id_map = {}
        
        # 处理每张图像
        print("开始转换标注...")
        
        for img_idx, file_name in enumerate(file_names):
            # 计算在哪个批次和批次内位置
            batch_idx = img_idx // statistics['batch_size']
            in_batch_idx = img_idx % statistics['batch_size']
            
            if batch_idx >= len(batch_images):
                break
            
            # 获取图像信息 - 使用固定的320x320尺寸
            image_info = self._get_original_image_info(file_name)
            image_id = image_info["id"]
            image_id_map[img_idx] = image_id
            
            # 使用固定的320x320尺寸，因为所有标注都是基于这个尺寸的
            target_height = 320
            target_width = 320
            
            coco_data["images"].append(image_info)
            
            # 获取当前图像的标注
            current_masks = batch_masks[batch_idx][in_batch_idx]  # [H, W, max_instances]
            current_boxes = batch_gt_boxes[batch_idx][in_batch_idx]  # [max_instances, 4]
            current_labels = batch_labels[batch_idx][in_batch_idx]  # [max_instances]
            
            # 处理每个实例
            for inst_idx in range(len(current_labels)):
                label = current_labels[inst_idx]
                
                # 跳过背景类和填充
                if label <= 0:
                    continue
                
                bbox = current_boxes[inst_idx]
                mask = current_masks[:, :, inst_idx]
                
                # 检查边界框有效性
                ymin, xmin, ymax, xmax = bbox
                if ymax <= ymin or xmax <= xmin:
                    continue
                
                # 如果使用mini mask，需要解码到320x320尺寸
                if statistics['use_mini_mask']:
                    # 使用固定的320x320尺寸
                    full_mask = self._decode_mini_mask(mask, bbox, 
                                                     (target_height, target_width))
                else:
                    # 如果不是mini mask，需要调整到320x320尺寸
                    if mask.shape != (target_height, target_width):
                        # 调整mask尺寸到320x320
                        mask_resized = cv2.resize(mask.astype(np.float32), 
                                                (target_width, target_height), 
                                                interpolation=cv2.INTER_NEAREST)
                        full_mask = (mask_resized >= 0.5).astype(np.uint8)
                    else:
                        full_mask = mask
                
                # 检查mask是否有效
                if np.sum(full_mask) == 0:
                    continue
                
                # 转换为RLE格式
                rle = self._mask_to_rle(full_mask)
                
                # 计算面积
                area = float(np.sum(full_mask))
                
                # 转换边界框格式 [ymin, xmin, ymax, xmax] -> [xmin, ymin, width, height]
                coco_bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                
                # 创建标注
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": coco_bbox,
                    "area": area,
                    "segmentation": rle,
                    "iscrowd": 0
                }
                
                coco_data["annotations"].append(annotation)
                annotation_id += 1
            
            if (img_idx + 1) % 100 == 0:
                print(f"  已处理 {img_idx + 1}/{len(file_names)} 张图像")
        
        # 复制图像文件
        self._copy_images_to_output(file_names, split)
        
        # 保存COCO格式标注
        output_file = os.path.join(self.output_dir, "annotations", f"instances_{split}2012.json")
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n✅ {split} 数据转换完成!")
        print(f"  输出文件: {output_file}")
        print(f"  图像数量: {len(coco_data['images'])}")
        print(f"  标注数量: {len(coco_data['annotations'])}")
        print(f"  类别数量: {len(coco_data['categories'])}")
        
        return output_file
    
    def convert_all(self):
        """转换所有数据"""
        print("=" * 60)
        print("开始转换所有标注数据为COCO格式")
        print("=" * 60)
        
        results = {}
        
        # 转换训练集
        try:
            train_file = self.convert_split('train')
            results['train'] = train_file
        except Exception as e:
            print(f"❌ 训练集转换失败: {e}")
            results['train'] = None
        
        # 转换验证集
        try:
            val_file = self.convert_split('val')
            results['val'] = val_file
        except Exception as e:
            print(f"❌ 验证集转换失败: {e}")
            results['val'] = None
        
        # 生成数据集配置信息
        self._generate_dataset_info()
        
        print("\n" + "=" * 60)
        print("🎉 转换完成!")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print("文件结构:")
        print("  annotations/")
        print("    ├── instances_train2012.json")
        print("    └── instances_val2012.json")
        print("  images/")
        print("    ├── 2007_000027.jpg")
        print("    └── ...")
        
        return results
    
    def _generate_dataset_info(self):
        """生成数据集配置信息"""
        dataset_info = {
            "dataset_name": "VOC2012_from_pickle",
            "description": "VOC2012 dataset converted from pickle format annotations",
            "classes": self.classes[1:],  # 不包含背景类
            "num_classes": len(self.classes) - 1,
            "data_root": self.output_dir,
            "ann_file": {
                "train": "annotations/instances_train2012.json",
                "val": "annotations/instances_val2012.json"
            },
            "img_prefix": "images/",
            "usage": {
                "mmdetection_config": {
                    "data_root": self.output_dir,
                    "train_ann_file": "annotations/instances_train2012.json",
                    "train_img_prefix": "images/",
                    "val_ann_file": "annotations/instances_val2012.json", 
                    "val_img_prefix": "images/",
                    "test_ann_file": "annotations/instances_val2012.json",
                    "test_img_prefix": "images/"
                }
            }
        }
        
        info_file = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📋 数据集信息保存到: {info_file}")


def main():
    """主函数"""
    print("🔄 标注格式转换器 - Pickle to COCO")
    
    # 检查输入文件
    annotation_dir = "./mask_rcnn_in_tf2_keras/final_annotations"
    if not os.path.exists(annotation_dir):
        print(f"❌ 标注目录不存在: {annotation_dir}")
        return
    
    train_file = os.path.join(annotation_dir, "annotations_train.pkl")
    val_file = os.path.join(annotation_dir, "annotations_val.pkl")
    
    if not os.path.exists(train_file):
        print(f"❌ 训练集标注文件不存在: {train_file}")
        return
    
    if not os.path.exists(val_file):
        print(f"❌ 验证集标注文件不存在: {val_file}")
        return
    
    print(f"✅ 找到标注文件:")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    
    try:
        # 创建转换器
        converter = AnnotationToCOCOConverter()
        
        # 执行转换
        results = converter.convert_all()
        
        if results['train'] and results['val']:
            print("\n💡 下一步:")
            print("1. 更新MMDetection配置文件以使用新的COCO格式数据")
            print("2. 运行: python update_config_for_coco.py")
            print("3. 开始训练: python train_models.py --model mask_rcnn")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 