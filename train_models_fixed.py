#!/usr/bin/env python3
"""
训练Mask R-CNN和Sparse R-CNN模型的脚本（修复版）
支持Tensorboard可视化训练过程
使用简化配置文件避免继承语法问题
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_gpu_count():
    """获取可用GPU数量"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_devices:
            return len(cuda_devices.split(','))
        else:
            return 0
    else:
        # 如果没有设置CUDA_VISIBLE_DEVICES，尝试检测所有可用GPU
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
        except FileNotFoundError:
            pass
        return 1  # 默认假设有1张GPU

def update_config_for_gpus(config_file, gpu_count):
    """根据GPU数量更新配置文件中的学习率"""
    print(f"检测到 {gpu_count} 张GPU，调整配置...")
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原始配置
    backup_file = config_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 根据GPU数量调整学习率
    if 'sparse_rcnn' in config_file:
        # Sparse R-CNN使用AdamW优化器
        base_lr = 0.000025
        new_lr = base_lr * gpu_count
        # 更新40个epoch的配置
        content = content.replace(
            f'lr=2.5e-05 * 4',  # 原来的4卡配置
            f'lr={base_lr} * {gpu_count}'
        )
        print(f"Sparse R-CNN学习率调整为: {new_lr} (40个epoch)")
    else:
        # Mask R-CNN使用SGD优化器
        base_lr = 0.02
        new_lr = base_lr * gpu_count
        # 更新40个epoch的配置
        content = content.replace(
            f'lr=0.02 * 8',  # 原来的8卡配置
            f'lr={base_lr} * {gpu_count}'
        )
        print(f"Mask R-CNN学习率调整为: {new_lr} (40个epoch)")
    
    # 写回配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)

def run_command(cmd, check=True):
    """运行shell命令"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        print(f"标准输出: {result.stdout}")
        return False
    print(f"命令执行成功: {result.stdout}")
    return True

def train_mask_rcnn():
    """训练Mask R-CNN模型"""
    print("开始训练Mask R-CNN...")
    
    config_file = "configs/mask_rcnn_voc_simple.py"  # 使用简化配置文件
    work_dir = "work_dirs/mask_rcnn"
    
    # 创建工作目录
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取GPU数量并更新配置
    gpu_count = get_gpu_count()
    update_config_for_gpus(config_file, gpu_count)
    
    # 根据GPU数量选择训练命令
    if gpu_count > 1:
        # 多卡分布式训练
        cmd = f"python -m torch.distributed.launch --nproc_per_node={gpu_count} mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir} --launcher pytorch"
    else:
        # 单卡训练
        cmd = f"python mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir}"
    
    return run_command(cmd)

def train_sparse_rcnn():
    """训练Sparse R-CNN模型"""
    print("开始训练Sparse R-CNN...")
    
    config_file = "sparse_rcnn_config_simple.py"  # 使用简化配置文件
    work_dir = "work_dirs/sparse_rcnn"
    
    # 创建工作目录
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取GPU数量并更新配置
    gpu_count = get_gpu_count()
    update_config_for_gpus(config_file, gpu_count)
    
    # 根据GPU数量选择训练命令
    if gpu_count > 1:
        # 多卡分布式训练
        cmd = f"python -m torch.distributed.launch --nproc_per_node={gpu_count} mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir} --launcher pytorch"
    else:
        # 单卡训练
        cmd = f"python mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir}"
    
    return run_command(cmd)

def test_model(config_file, checkpoint_file, model_name):
    """使用最佳权重在VOC2012验证集上测试模型"""
    print(f"使用最佳权重测试{model_name}模型...")
    print(f"配置文件: {config_file}")
    print(f"权重文件: {checkpoint_file}")
    print(f"测试集: VOC2012验证集 (5824张图片)")
    
    result_file = f"results/{model_name}_final_results.pkl"
    
    # 创建结果目录
    Path("results").mkdir(exist_ok=True)
    Path(f"visualizations/{model_name}").mkdir(parents=True, exist_ok=True)
    
    cmd = f"""
    python mmdetection_repo/tools/test.py {config_file} {checkpoint_file} \
        --out {result_file} \
        --eval bbox segm \
        --show-dir visualizations/{model_name} \
        --cfg-options test_evaluator.classwise=True
    """
    
    success = run_command(cmd)
    if success:
        print(f"✅ {model_name}测试完成!")
        print(f"📊 结果保存在: {result_file}")
        print(f"🖼️  可视化结果保存在: visualizations/{model_name}/")
    
    return success

def start_tensorboard_info():
    """显示Tensorboard启动信息"""
    gpu_count = get_gpu_count()
    print("\n" + "="*60)
    print("📊 Tensorboard 可视化说明")
    print("="*60)
    print(f"🔧 检测到 {gpu_count} 张GPU卡")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"🎯 使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print()
    print("📋 数据集配置:")
    print("  🗂️  训练集: VOC2012 train.txt (5718张图片)")
    print("  🗂️  验证集: VOC2012 val.txt (5824张图片)")
    print("  📅 训练轮次: 40个epoch")
    print("  ✅ 每个epoch进行验证")
    print("  🏆 基于验证mAP保存最佳模型")
    print()
    print("🔧 配置文件修复:")
    print("  ✅ 使用简化配置文件避免继承语法问题")
    print("  📁 Mask R-CNN: configs/mask_rcnn_voc_simple.py")
    print("  📁 Sparse R-CNN: sparse_rcnn_config_simple.py")
    print()
    print("训练开始后，您可以使用以下命令启动Tensorboard:")
    print("  python start_tensorboard.py")
    print()
    print("或者手动启动:")
    print("  tensorboard --logdir=work_dirs --port=6006")
    print()
    print("然后在浏览器中访问: http://localhost:6006")
    print()
    print("🎯 分离的可视化内容包括:")
    print("  📈 Train_Loss/: 训练过程中的各种loss曲线")
    print("  📉 Val_Loss/: 验证过程中的各种loss曲线")
    print("  📊 Val_mAP/: 验证集的mAP和AP曲线")
    print("  🔧 Learning_Rate/: 学习率变化曲线")
    print("  ⏱️  Time/: 训练和验证时间统计")
    print("  🖼️  Val_Metrics/: 其他验证指标")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='训练和测试目标检测模型（修复版）')
    parser.add_argument('--model', choices=['mask_rcnn', 'sparse_rcnn', 'both'], 
                       default='both', help='选择要训练的模型')
    parser.add_argument('--test-only', action='store_true', 
                       help='只进行测试，不训练')
    parser.add_argument('--start-tensorboard', action='store_true',
                       help='训练完成后自动启动Tensorboard')
    parser.add_argument('--no-interactive', action='store_true',
                       help='非交互模式，跳过用户输入（适用于nohup等后台运行）')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    dirs = ["work_dirs", "results", "visualizations"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # 显示Tensorboard信息
    if not args.test_only:
        start_tensorboard_info()
        if not args.no_interactive:
            input("按Enter键继续训练...")
        else:
            print("非交互模式，自动开始训练...")
    
    if not args.test_only:
        # 训练模型
        if args.model in ['mask_rcnn', 'both']:
            success = train_mask_rcnn()
            if not success:
                print("Mask R-CNN训练失败")
                return
        
        if args.model in ['sparse_rcnn', 'both']:
            success = train_sparse_rcnn()
            if not success:
                print("Sparse R-CNN训练失败")
                return
    
    # 使用最佳权重测试模型
    print("\n🏆 使用最佳权重进行最终测试...")
    
    if args.model in ['mask_rcnn', 'both']:
        # 查找最佳权重文件
        if os.path.exists("work_dirs/mask_rcnn/"):
            # 查找best开头的权重文件
            best_files = [f for f in os.listdir("work_dirs/mask_rcnn/") if f.startswith("best_") and f.endswith(".pth")]
            if best_files:
                mask_rcnn_checkpoint = os.path.join("work_dirs/mask_rcnn/", best_files[0])
                print(f"🎯 找到Mask R-CNN最佳权重: {best_files[0]}")
            elif os.path.exists("work_dirs/mask_rcnn/latest.pth"):
                mask_rcnn_checkpoint = "work_dirs/mask_rcnn/latest.pth"
                print(f"⚠️  未找到最佳权重，使用最新权重: latest.pth")
            else:
                print("❌ Mask R-CNN检查点文件不存在")
                mask_rcnn_checkpoint = None
                
            if mask_rcnn_checkpoint:
                test_model("configs/mask_rcnn_voc_simple.py", mask_rcnn_checkpoint, "mask_rcnn")
    
    if args.model in ['sparse_rcnn', 'both']:
        # 查找最佳权重文件
        if os.path.exists("work_dirs/sparse_rcnn/"):
            # 查找best开头的权重文件
            best_files = [f for f in os.listdir("work_dirs/sparse_rcnn/") if f.startswith("best_") and f.endswith(".pth")]
            if best_files:
                sparse_rcnn_checkpoint = os.path.join("work_dirs/sparse_rcnn/", best_files[0])
                print(f"🎯 找到Sparse R-CNN最佳权重: {best_files[0]}")
            elif os.path.exists("work_dirs/sparse_rcnn/latest.pth"):
                sparse_rcnn_checkpoint = "work_dirs/sparse_rcnn/latest.pth"
                print(f"⚠️  未找到最佳权重，使用最新权重: latest.pth")
            else:
                print("❌ Sparse R-CNN检查点文件不存在")
                sparse_rcnn_checkpoint = None
                
            if sparse_rcnn_checkpoint:
                test_model("sparse_rcnn_config_simple.py", sparse_rcnn_checkpoint, "sparse_rcnn")
    
    print("训练和测试完成!")
    
    # 可选择自动启动Tensorboard
    if args.start_tensorboard:
        print("\n启动Tensorboard...")
        subprocess.run([sys.executable, "start_tensorboard.py"], cwd=".")

if __name__ == "__main__":
    main() 