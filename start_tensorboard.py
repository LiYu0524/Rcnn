 #!/usr/bin/env python3
"""
启动Tensorboard可视化训练过程的脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def start_tensorboard(log_dir="work_dirs", port=6006, host="0.0.0.0"):
    """启动Tensorboard服务"""
    
    # 检查日志目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误: 日志目录 {log_dir} 不存在")
        print("请先运行训练脚本生成日志文件")
        return False
    
    # 查找所有可用的日志目录
    log_dirs = []
    for root, dirs, files in os.walk(log_dir):
        # 查找包含Tensorboard日志的目录
        if any(f.startswith('events.out.tfevents') for f in files):
            log_dirs.append(root)
    
    if not log_dirs:
        print(f"在 {log_dir} 中未找到Tensorboard日志文件")
        print("请确保训练已经开始并生成了日志文件")
        return False
    
    print(f"找到以下日志目录:")
    for i, dir_path in enumerate(log_dirs):
        print(f"  {i+1}. {dir_path}")
    
    # 启动Tensorboard
    cmd = f"tensorboard --logdir={log_dir} --port={port} --host={host}"
    print(f"\n启动Tensorboard...")
    print(f"命令: {cmd}")
    print(f"访问地址: http://{host}:{port}")
    print("按 Ctrl+C 停止Tensorboard服务")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\nTensorboard服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"启动Tensorboard失败: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='启动Tensorboard可视化训练过程')
    parser.add_argument('--log-dir', default='work_dirs', 
                       help='日志目录路径 (默认: work_dirs)')
    parser.add_argument('--port', type=int, default=6006, 
                       help='Tensorboard端口 (默认: 6006)')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Tensorboard主机地址 (默认: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("=== Tensorboard 可视化工具 ===")
    print(f"日志目录: {args.log_dir}")
    print(f"端口: {args.port}")
    print(f"主机: {args.host}")
    print()
    
    # 检查tensorboard是否安装
    try:
        subprocess.run(['tensorboard', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到tensorboard命令")
        print("请安装tensorboard: pip install tensorboard")
        return
    
    success = start_tensorboard(args.log_dir, args.port, args.host)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()