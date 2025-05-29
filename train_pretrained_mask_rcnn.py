#!/usr/bin/env python3
"""
使用COCO预训练模型微调Mask R-CNN在VOC2012数据集上
支持Tensorboard可视化和最佳模型保存
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import glob

def check_pretrained_weights():
    """检查预训练权重文件是否存在"""
    pretrain_path = "pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
    
    if not os.path.exists(pretrain_path):
        print(f"❌ 预训练权重文件不存在: {pretrain_path}")
        print("\n💡 请确保预训练权重文件位于正确位置:")
        print(f"   {os.path.abspath(pretrain_path)}")
        print("\n📥 您可以从以下位置复制:")
        print("   /mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth")
        print("\n🔧 或者运行以下命令:")
        print(f"   mkdir -p pretrain")
        print(f"   cp /mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth pretrain/")
        return False
    
    print(f"✅ 找到预训练权重: {pretrain_path}")
    return True

def get_gpu_count():
    """获取可用GPU数量"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_devices:
            return len(cuda_devices.split(','))
        else:
            return 0
    else:
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
        except FileNotFoundError:
            pass
        return 1

def update_config_for_gpus(config_file, gpu_count):
    """根据GPU数量更新配置文件中的学习率"""
    print(f"🔧 检测到 {gpu_count} 张GPU，调整微调学习率...")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原始配置
    backup_file = config_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 微调学习率策略：基础学习率 * GPU数量
    base_lr = 0.002  # 微调基础学习率
    new_lr = base_lr * gpu_count
    
    # 更新学习率
    content = content.replace(
        f'lr=0.002,  # 微调学习率，比从头训练小10倍',
        f'lr={new_lr},  # 微调学习率，适配{gpu_count}张GPU'
    )
    
    # 写回配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📊 微调学习率设置:")
    print(f"   基础学习率: {base_lr}")
    print(f"   GPU数量: {gpu_count}")
    print(f"   最终学习率: {new_lr}")
    print(f"   Backbone学习率: {new_lr * 0.1}")
    print(f"   Neck学习率: {new_lr * 0.5}")

def run_command(cmd, check=True):
    """运行shell命令"""
    print(f"🚀 执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ 命令执行失败: {result.stderr}")
        print(f"📋 标准输出: {result.stdout}")
        return False
    print(f"✅ 命令执行成功")
    if result.stdout.strip():
        print(f"📋 输出: {result.stdout.strip()}")
    return True

def copy_pretrained_weights():
    """复制预训练权重到本地"""
    source_path = "/mnt/data/liyu/mm/pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
    target_dir = "pretrain"
    target_path = os.path.join(target_dir, "mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth")
    
    if os.path.exists(target_path):
        print(f"✅ 预训练权重已存在: {target_path}")
        return True
    
    if not os.path.exists(source_path):
        print(f"❌ 源预训练权重文件不存在: {source_path}")
        return False
    
    # 创建目录
    Path(target_dir).mkdir(exist_ok=True)
    
    # 复制文件
    print(f"📥 复制预训练权重...")
    print(f"   从: {source_path}")
    print(f"   到: {target_path}")
    
    cmd = f"cp {source_path} {target_path}"
    return run_command(cmd)

def train_pretrained_mask_rcnn():
    """训练预训练Mask R-CNN模型"""
    print("🎯 开始使用COCO预训练模型微调Mask R-CNN...")
    
    # 使用简化的配置文件
    config_file = "configs/mask_rcnn_voc_pretrained_simple.py"
    work_dir = "work_dirs/mask_rcnn_pretrained"
    
    # 检查配置文件
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 检查并复制预训练权重
    if not copy_pretrained_weights():
        return False
    
    # 创建工作目录
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取GPU数量并更新配置
    gpu_count = get_gpu_count()
    update_config_for_gpus(config_file, gpu_count)
    
    # 设置PYTHONPATH以包含当前目录（包含custom_hooks.py）
    current_dir = os.path.abspath(".")
    env_vars = f"PYTHONPATH={current_dir}:$PYTHONPATH"
    
    # 根据GPU数量选择训练命令
    if gpu_count > 1:
        cmd = f"{env_vars} python -m torch.distributed.launch --nproc_per_node={gpu_count} --master_port=29501 mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir} --launcher pytorch"
    else:
        cmd = f"{env_vars} python mmdetection_repo/tools/train.py {config_file} --work-dir {work_dir}"
    
    return run_command(cmd)

def find_best_checkpoint(work_dir):
    """查找最佳检查点文件"""
    # 查找best_开头的权重文件
    best_files = glob.glob(os.path.join(work_dir, "best_*.pth"))
    if best_files:
        # 按修改时间排序，返回最新的
        best_files.sort(key=os.path.getmtime, reverse=True)
        return best_files[0]
    
    # 如果没有best文件，查找latest.pth
    latest_file = os.path.join(work_dir, "latest.pth")
    if os.path.exists(latest_file):
        return latest_file
    
    # 查找epoch_xx.pth文件
    epoch_files = glob.glob(os.path.join(work_dir, "epoch_*.pth"))
    if epoch_files:
        # 按epoch数字排序，返回最大的
        epoch_files.sort(key=lambda x: int(x.split('epoch_')[1].split('.pth')[0]), reverse=True)
        return epoch_files[0]
    
    return None

def test_pretrained_model():
    """测试预训练微调后的模型"""
    print("🧪 开始测试预训练微调后的Mask R-CNN模型...")
    
    # 使用简化的配置文件
    config_file = "configs/mask_rcnn_voc_pretrained_simple.py"
    work_dir = "work_dirs/mask_rcnn_pretrained"
    
    # 查找最佳检查点
    checkpoint_file = find_best_checkpoint(work_dir)
    if not checkpoint_file:
        print(f"❌ 在 {work_dir} 中找不到检查点文件")
        return False
    
    print(f"📋 使用检查点: {checkpoint_file}")
    print(f"📋 配置文件: {config_file}")
    print(f"📋 测试集: VOC2012验证集 (5824张图片)")
    
    # 创建结果目录
    result_file = "results/mask_rcnn_pretrained_final_results.pkl"
    vis_dir = "visualizations/mask_rcnn_pretrained"
    
    Path("results").mkdir(exist_ok=True)
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    
    # 测试命令
    cmd = f"""
    python mmdetection_repo/tools/test.py {config_file} {checkpoint_file} \
        --out {result_file} \
        --eval bbox segm \
        --show-dir {vis_dir} \
        --cfg-options test_evaluator.classwise=True
    """
    
    success = run_command(cmd)
    if success:
        print(f"✅ 预训练模型测试完成!")
        print(f"📊 结果保存在: {result_file}")
        print(f"🖼️  可视化结果保存在: {vis_dir}/")
    
    return success

def show_tensorboard_info():
    """显示Tensorboard信息"""
    gpu_count = get_gpu_count()
    print("\n" + "="*70)
    print("📊 Tensorboard 可视化说明 - 预训练模型微调")
    print("="*70)
    print(f"🔧 检测到 {gpu_count} 张GPU卡")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"🎯 使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print()
    print("🎯 微调配置:")
    print("  📦 预训练模型: Mask R-CNN R50-FPN (COCO)")
    print("  🔄 微调策略: 重新初始化分类头和掩码头")
    print("  📚 学习率策略: 分层学习率 (backbone更小)")
    print("  📅 训练轮次: 40个epoch")
    print("  ✅ 每个epoch进行验证")
    print("  🏆 基于验证mAP保存最佳模型")
    print()
    print("📋 数据集配置:")
    print("  🗂️  训练集: VOC2012 train.txt (5718张图片)")
    print("  🗂️  验证集: VOC2012 val.txt (5824张图片)")
    print("  🎨 数据增强: PhotoMetricDistortion")
    print()
    print("训练开始后，您可以使用以下命令启动Tensorboard:")
    print("  tensorboard --logdir=work_dirs/mask_rcnn_pretrained/tensorboard_logs --port=6006")
    print()
    print("然后在浏览器中访问: http://localhost:6006")
    print()
    print("🎯 分离的可视化内容包括:")
    print("  📈 Train_Loss/: 训练过程中的各种loss曲线")
    print("  📉 Val_Loss/: 验证过程中的各种loss曲线") 
    print("  📊 Val_mAP/: 验证集的mAP和AP曲线")
    print("  🔧 Learning_Rate/: 分层学习率变化曲线")
    print("  ⏱️  Time/: 训练和验证时间统计")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='使用COCO预训练模型微调Mask R-CNN')
    parser.add_argument('--train', action='store_true', 
                       help='训练模型')
    parser.add_argument('--test', action='store_true', 
                       help='测试模型')
    parser.add_argument('--both', action='store_true', 
                       help='训练并测试模型')
    parser.add_argument('--check-weights', action='store_true',
                       help='检查预训练权重')
    
    args = parser.parse_args()
    
    if not any([args.train, args.test, args.both, args.check_weights]):
        args.both = True  # 默认训练并测试
    
    print("🎯 Mask R-CNN 预训练模型微调脚本")
    print("="*50)
    
    if args.check_weights:
        check_pretrained_weights()
        return
    
    # 显示Tensorboard信息
    show_tensorboard_info()
    
    success = True
    
    if args.train or args.both:
        print("\n🚀 开始训练...")
        success = train_pretrained_mask_rcnn()
        
        if success:
            print("\n✅ 训练完成!")
        else:
            print("\n❌ 训练失败!")
            return
    
    if args.test or args.both:
        if success:
            print("\n🧪 开始测试...")
            test_success = test_pretrained_model()
            
            if test_success:
                print("\n🎉 所有任务完成!")
                print("\n💡 下一步:")
                print("1. 查看Tensorboard可视化结果")
                print("2. 检查测试结果文件")
                print("3. 查看可视化图像")
            else:
                print("\n❌ 测试失败!")

if __name__ == "__main__":
    main() 