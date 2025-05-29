#!/usr/bin/env python3
"""
Mask R-CNN 和 Sparse R-CNN 项目入口脚本
"""

import sys
import os
from pathlib import Path

def main():
    print("🎯 Mask R-CNN 和 Sparse R-CNN 在VOC数据集上的训练与测试")
    print("=" * 60)
    
    print("\n📋 项目功能：")
    print("1. 在VOC数据集上训练Mask R-CNN和Sparse R-CNN")
    print("2. 可视化Mask R-CNN的proposal box和最终结果")
    print("3. 对比两个模型的检测和分割结果")
    print("4. 在外部图像上测试模型泛化能力")
    
    print("\n🚀 快速开始：")
    print("python run_experiment.py                    # 运行完整实验")
    print("python run_experiment.py --model mask_rcnn  # 只训练Mask R-CNN")
    print("python run_experiment.py --step setup       # 只设置环境")
    print("python run_experiment.py --skip-train       # 跳过训练，直接可视化")
    
    print("\n📁 主要文件：")
    print("- run_experiment.py      # 主运行脚本")
    print("- setup_environment.py   # 环境设置")
    print("- train_models.py        # 模型训练")
    print("- visualize_results.py   # 结果可视化")
    print("- README.md              # 详细说明文档")
    
    print("\n💡 使用建议：")
    print("1. 首次运行请使用: python run_experiment.py")
    print("2. 确保有足够的GPU内存（推荐8GB+）")
    print("3. 完整训练需要几个小时，请耐心等待")
    print("4. 查看README.md获取详细说明")
    
    print("\n" + "=" * 60)
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            # 运行主实验
            os.system("python run_experiment.py")
        elif sys.argv[1] == "help":
            # 显示帮助
            os.system("python run_experiment.py --help")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 'python mask_rcnn.py run' 开始实验")
            print("使用 'python mask_rcnn.py help' 查看帮助")
    else:
        print("使用 'python mask_rcnn.py run' 开始实验")
        print("或直接运行 'python run_experiment.py'")

if __name__ == "__main__":
    main()
