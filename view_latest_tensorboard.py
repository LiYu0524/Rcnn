#!/usr/bin/env python3
"""
查看最新训练的Tensorboard可视化脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import glob
from datetime import datetime

def find_latest_training_logs():
    """查找最新的训练日志目录"""
    
    # 可能的日志目录
    possible_dirs = [
        'work_dirs/sparse_rcnn',
        'work_dirs/mask_rcnn',
        'work_dirs'
    ]
    
    latest_logs = []
    
    for base_dir in possible_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # 查找所有包含tensorboard日志的子目录
        for root, dirs, files in os.walk(base_dir):
            # 查找events文件或tensorboard_logs目录
            has_events = any(f.startswith('events.out.tfevents') for f in files)
            has_tb_logs = 'tensorboard_logs' in dirs
            
            if has_events or has_tb_logs:
                # 获取目录的修改时间
                try:
                    mtime = os.path.getmtime(root)
                    latest_logs.append((root, mtime))
                except OSError:
                    continue
    
    if not latest_logs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_logs.sort(key=lambda x: x[1], reverse=True)
    return latest_logs

def start_tensorboard_for_latest(port=6006, host="0.0.0.0"):
    """为最新的训练启动Tensorboard"""
    
    print("🔍 查找最新的训练日志...")
    latest_logs = find_latest_training_logs()
    
    if not latest_logs:
        print("❌ 未找到任何训练日志")
        print("请确保已经开始训练并生成了日志文件")
        return False
    
    print(f"📊 找到 {len(latest_logs)} 个训练日志目录:")
    for i, (log_dir, mtime) in enumerate(latest_logs[:5]):  # 显示最新的5个
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {log_dir} (修改时间: {time_str})")
    
    # 使用最新的日志目录
    latest_dir = latest_logs[0][0]
    print(f"\n🎯 使用最新的日志目录: {latest_dir}")
    
    # 检查是否有tensorboard_logs子目录
    tb_logs_dir = os.path.join(latest_dir, 'tensorboard_logs')
    if os.path.exists(tb_logs_dir):
        log_dir = tb_logs_dir
        print(f"📁 发现专用Tensorboard目录: {tb_logs_dir}")
    else:
        log_dir = latest_dir
    
    # 启动Tensorboard
    cmd = f"tensorboard --logdir={log_dir} --port={port} --host={host}"
    print(f"\n🚀 启动Tensorboard...")
    print(f"📝 命令: {cmd}")
    print(f"🌐 访问地址: http://localhost:{port}")
    print(f"🌐 远程访问: http://{host}:{port}")
    print("\n📊 可视化内容包括:")
    print("  📈 Train_Loss/: 训练过程中的各种loss曲线")
    print("  📉 Val_Loss/: 验证过程中的各种loss曲线") 
    print("  📊 Val_mAP/: 验证集的mAP和AP曲线")
    print("  🔧 Learning_Rate/: 学习率变化曲线")
    print("  ⏱️  Time/: 训练和验证时间统计")
    print("\n按 Ctrl+C 停止Tensorboard服务")
    print("="*60)
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n✅ Tensorboard服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动Tensorboard失败: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='查看最新训练的Tensorboard可视化')
    parser.add_argument('--port', type=int, default=6006, 
                       help='Tensorboard端口 (默认: 6006)')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Tensorboard主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--list', action='store_true',
                       help='只列出可用的日志目录，不启动Tensorboard')
    
    args = parser.parse_args()
    
    print("=== 最新训练 Tensorboard 可视化工具 ===")
    print(f"端口: {args.port}")
    print(f"主机: {args.host}")
    print()
    
    if args.list:
        # 只列出可用的日志目录
        latest_logs = find_latest_training_logs()
        if not latest_logs:
            print("❌ 未找到任何训练日志")
        else:
            print(f"📊 找到 {len(latest_logs)} 个训练日志目录:")
            for i, (log_dir, mtime) in enumerate(latest_logs):
                time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {i+1}. {log_dir} (修改时间: {time_str})")
        return
    
    # 检查tensorboard是否安装
    try:
        result = subprocess.run(['tensorboard', '--version'], 
                              capture_output=True, check=True, text=True)
        print(f"✅ Tensorboard版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: 未找到tensorboard命令")
        print("请安装tensorboard: pip install tensorboard")
        return
    
    success = start_tensorboard_for_latest(args.port, args.host)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 