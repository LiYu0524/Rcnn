#!/usr/bin/env python3
"""
自定义钩子，用于分离记录训练loss、验证loss、验证mAP到Tensorboard
"""

import os
from typing import Optional, Sequence
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS


@HOOKS.register_module()
class CustomLoggerHook(Hook):
    """自定义日志钩子，分离记录训练和验证指标到Tensorboard"""
    
    def __init__(self,
                 log_metric_by_epoch: bool = True,
                 log_train_loss: bool = True,
                 log_val_loss: bool = True,
                 log_val_map: bool = True,
                 interval: int = 50,
                 tensorboard_tag_prefix: str = ""):
        """
        Args:
            log_metric_by_epoch: 是否按epoch记录指标
            log_train_loss: 是否记录训练loss
            log_val_loss: 是否记录验证loss
            log_val_map: 是否记录验证mAP
            interval: 训练loss记录间隔
            tensorboard_tag_prefix: Tensorboard标签前缀
        """
        self.log_metric_by_epoch = log_metric_by_epoch
        self.log_train_loss = log_train_loss
        self.log_val_loss = log_val_loss
        self.log_val_map = log_val_map
        self.interval = interval
        self.tensorboard_tag_prefix = tensorboard_tag_prefix
        
    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch = None,
                        outputs = None) -> None:
        """训练迭代后记录训练loss"""
        if not self.log_train_loss:
            return
            
        # 每隔interval次迭代记录一次
        if (runner.iter + 1) % self.interval != 0:
            return
            
        # 获取训练loss
        if hasattr(runner, 'log_buffer') and runner.log_buffer.ready:
            log_dict = runner.log_buffer.output
            
            # 记录各种训练loss到Tensorboard
            for key, value in log_dict.items():
                if 'loss' in key.lower() and isinstance(value, (int, float)):
                    # 分离记录到训练loss组
                    prefix = f"{self.tensorboard_tag_prefix}/" if self.tensorboard_tag_prefix else ""
                    tag = f"{prefix}Train_Loss/{key}"
                    runner.visualizer.add_scalar(tag, value, runner.iter)
                    
    def after_val_epoch(self,
                       runner: Runner,
                       metrics = None) -> None:
        """验证epoch后记录验证指标"""
        if metrics is None:
            return
            
        current_epoch = runner.epoch
        prefix = f"{self.tensorboard_tag_prefix}/" if self.tensorboard_tag_prefix else ""
        
        # 记录验证loss
        if self.log_val_loss:
            for key, value in metrics.items():
                if 'loss' in key.lower() and isinstance(value, (int, float)):
                    tag = f"{prefix}Val_Loss/{key}"
                    runner.visualizer.add_scalar(tag, value, current_epoch)
        
        # 记录验证mAP
        if self.log_val_map:
            for key, value in metrics.items():
                if 'map' in key.lower() or 'ap' in key.lower():
                    if isinstance(value, (int, float)):
                        tag = f"{prefix}Val_mAP/{key}"
                        runner.visualizer.add_scalar(tag, value, current_epoch)
                        
        # 记录其他验证指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and 'loss' not in key.lower() and 'map' not in key.lower() and 'ap' not in key.lower():
                tag = f"{prefix}Val_Metrics/{key}"
                runner.visualizer.add_scalar(tag, value, current_epoch)


@HOOKS.register_module()
class TensorboardLoggerHook(Hook):
    """增强的Tensorboard日志钩子"""
    
    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 50,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        """
        Args:
            log_dir: Tensorboard日志目录
            interval: 记录间隔
            ignore_last: 是否忽略最后一次记录
            reset_flag: 是否重置标志
            by_epoch: 是否按epoch记录
        """
        self.log_dir = log_dir
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch
        
    def before_run(self, runner: Runner) -> None:
        """运行前初始化"""
        if self.log_dir is None:
            self.log_dir = os.path.join(runner.work_dir, 'tensorboard_logs')
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch = None,
                        outputs = None) -> None:
        """训练迭代后记录"""
        if (runner.iter + 1) % self.interval != 0:
            return
            
        # 记录学习率
        if hasattr(runner, 'optim_wrapper'):
            for i, lr in enumerate(runner.optim_wrapper.get_lr()['lr']):
                runner.visualizer.add_scalar(f'Learning_Rate/lr_{i}', lr, runner.iter)
                
        # 记录训练时间
        if hasattr(runner, 'log_buffer') and runner.log_buffer.ready:
            log_dict = runner.log_buffer.output
            if 'time' in log_dict:
                runner.visualizer.add_scalar('Time/iter_time', log_dict['time'], runner.iter)
                
    def after_val_epoch(self,
                       runner: Runner,
                       metrics = None) -> None:
        """验证后记录epoch级别的指标"""
        if metrics is None:
            return
            
        current_epoch = runner.epoch
        
        # 记录验证时间
        if 'val_time' in metrics:
            runner.visualizer.add_scalar('Time/val_time', metrics['val_time'], current_epoch) 