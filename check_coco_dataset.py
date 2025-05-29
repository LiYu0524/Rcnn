#!/usr/bin/env python3
import json
import os

def check_coco_dataset(data_root):
    """检查COCO格式数据集的基本信息"""
    
    # 检查训练集
    train_ann_file = os.path.join(data_root, 'annotations/instances_train2017.json')
    if os.path.exists(train_ann_file):
        with open(train_ann_file, 'r') as f:
            train_data = json.load(f)
        
        print("=== 训练集信息 ===")
        print(f"图片数量: {len(train_data['images'])}")
        print(f"标注数量: {len(train_data['annotations'])}")
        print(f"类别数量: {len(train_data['categories'])}")
        
        print("\n类别信息:")
        for cat in train_data['categories']:
            print(f"  ID: {cat['id']}, 名称: {cat['name']}")
    
    # 检查验证集
    val_ann_file = os.path.join(data_root, 'annotations/instances_val2017.json')
    if os.path.exists(val_ann_file):
        with open(val_ann_file, 'r') as f:
            val_data = json.load(f)
        
        print("\n=== 验证集信息 ===")
        print(f"图片数量: {len(val_data['images'])}")
        print(f"标注数量: {len(val_data['annotations'])}")
    
    # 检查图片目录
    train_img_dir = os.path.join(data_root, 'train2017')
    val_img_dir = os.path.join(data_root, 'val2017')
    
    if os.path.exists(train_img_dir):
        train_imgs = len([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
        print(f"\n训练集图片文件数量: {train_imgs}")
    
    if os.path.exists(val_img_dir):
        val_imgs = len([f for f in os.listdir(val_img_dir) if f.endswith('.jpg')])
        print(f"验证集图片文件数量: {val_imgs}")

if __name__ == "__main__":
    data_root = "/mnt/data/liyu/mm/data/coco"
    check_coco_dataset(data_root) 