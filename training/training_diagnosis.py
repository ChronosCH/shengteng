#!/usr/bin/env python3
"""
训练异常诊断脚本 - 分析WER保持0的原因
"""

import json
import os
from collections import defaultdict

def analyze_training_logs(log_dir):
    """分析训练日志文件"""
    
    print("=== 训练异常诊断 ===")
    
    # 查找日志文件
    log_files = []
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.endswith('.log') or 'train' in file.lower():
                log_files.append(os.path.join(log_dir, file))
    
    if not log_files:
        print(f"❌ 在 {log_dir} 中未找到日志文件")
        print(f"💡 这可能是因为:")
        print(f"  1. 训练日志没有保存到该目录")
        print(f"  2. 日志文件被移动或删除")
        print(f"  3. 训练使用了不同的输出目录")
        return
    
    print(f"发现 {len(log_files)} 个日志文件:")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    # 分析每个日志文件
    for log_file in log_files:
        print(f"\n=== 分析 {os.path.basename(log_file)} ===")
        analyze_single_log(log_file)

def analyze_single_log(log_file):
    """分析单个训练日志"""
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ 无法读取日志文件: {e}")
        return
    
    print(f"日志总行数: {len(lines)}")
    
    # 分析关键指标
    epochs = []
    wer_values = []
    loss_values = []
    best_model_updates = []
    
    current_epoch = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测epoch信息
        if 'epoch' in line.lower() or 'Epoch' in line:
            if 'Epoch:' in line or 'epoch:' in line:
                try:
                    epoch_part = line.split('Epoch:')[-1] if 'Epoch:' in line else line.split('epoch:')[-1]
                    epoch_num = int(epoch_part.split()[0].split('/')[0])
                    current_epoch = epoch_num
                    if epoch_num not in epochs:
                        epochs.append(epoch_num)
                except:
                    pass
        
        # 检测WER值
        if 'WER' in line:
            try:
                wer_part = line.split('WER')[-1]
                # 查找数字（可能是百分比或小数）
                import re
                wer_match = re.search(r'[:\s=]+([\d.]+)', wer_part)
                if wer_match:
                    wer_val = float(wer_match.group(1))
                    wer_values.append((current_epoch or len(wer_values), wer_val, line))
            except:
                pass
        
        # 检测loss值
        if 'loss' in line.lower() and ('=' in line or ':' in line):
            try:
                import re
                loss_match = re.search(r'loss[:\s=]+([\d.]+)', line.lower())
                if loss_match:
                    loss_val = float(loss_match.group(1))
                    loss_values.append((current_epoch or len(loss_values), loss_val, line))
            except:
                pass
        
        # 检测best model更新
        if 'best' in line.lower() and ('model' in line.lower() or 'checkpoint' in line.lower()):
            best_model_updates.append((current_epoch or i, line))
    
    # 报告分析结果
    print(f"\n📊 训练指标统计:")
    print(f"  检测到的epoch数: {len(epochs)} {epochs if epochs else ''}")
    print(f"  WER记录数: {len(wer_values)}")
    print(f"  Loss记录数: {len(loss_values)}")
    print(f"  Best model更新数: {len(best_model_updates)}")
    
    if wer_values:
        print(f"\n📈 WER变化趋势:")
        for epoch, wer, line in wer_values[:10]:  # 只显示前10个
            print(f"  Epoch {epoch}: WER={wer:.6f}")
            if wer == 0.0:
                print(f"    ⚠️  发现WER=0 - 这可能导致best model不更新")
        
        # 检查WER是否一直为0
        zero_wer_count = sum(1 for _, wer, _ in wer_values if wer == 0.0)
        if zero_wer_count > 1:
            print(f"    🚨 警告: {zero_wer_count}/{len(wer_values)} 次WER为0")
            print(f"    🔍 这解释了为什么训练在epoch 1后停止更新best model")
    
    if loss_values:
        print(f"\n📉 Loss变化趋势:")
        for epoch, loss, line in loss_values[:10]:  # 只显示前10个
            print(f"  Epoch {epoch}: Loss={loss:.6f}")
    
    if best_model_updates:
        print(f"\n💾 Best Model更新记录:")
        for epoch, line in best_model_updates:
            print(f"  Epoch {epoch}: {line.strip()}")
    else:
        print(f"\n❌ 未发现best model更新记录")
        print(f"  这确认了训练过程中best model没有被更新")

def analyze_model_checkpoints(checkpoint_dir):
    """分析模型检查点文件"""
    
    print(f"\n=== 检查点文件分析 ===")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 检查点目录不存在: {checkpoint_dir}")
        return
    
    # 列出所有检查点文件
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt') or 'epoch' in file.lower():
            checkpoint_files.append(file)
    
    print(f"发现 {len(checkpoint_files)} 个检查点文件:")
    
    # 按修改时间排序
    checkpoint_info = []
    for file in checkpoint_files:
        file_path = os.path.join(checkpoint_dir, file)
        stat = os.stat(file_path)
        checkpoint_info.append((file, stat.st_mtime, stat.st_size))
    
    checkpoint_info.sort(key=lambda x: x[1])  # 按时间排序
    
    for file, mtime, size in checkpoint_info:
        import datetime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_mb = size / (1024 * 1024)
        print(f"  {file}: {time_str}, {size_mb:.1f}MB")
    
    # 检查best model文件
    best_files = [f for f in checkpoint_files if 'best' in f.lower()]
    if best_files:
        print(f"\n🏆 Best model文件:")
        for file in best_files:
            file_path = os.path.join(checkpoint_dir, file)
            stat = os.stat(file_path)
            time_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {file}: {time_str}")
    else:
        print(f"\n❌ 未找到best model文件")
        print(f"  这确认了WER=0导致best model从未被保存")

def analyze_config_file(config_path):
    """分析训练配置文件"""
    
    print(f"\n=== 配置文件分析 ===")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✓ 成功读取配置文件: {config_path}")
        
        # 检查关键配置 - 需要在嵌套结构中查找
        training_config = config.get('training', {})
        
        key_configs = {
            'num_epochs': training_config.get('num_epochs'),
            'early_stopping_patience': training_config.get('early_stopping_patience'),
            'learning_rate': training_config.get('learning_rate'),
            'batch_size': training_config.get('batch_size'),
            'save_interval': training_config.get('save_interval'),
            'eval_interval': training_config.get('eval_interval')
        }
        
        print(f"\n🔧 关键训练配置:")
        for key, value in key_configs.items():
            if value is not None:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: 未设置")
        
        # 检查模型配置
        model_config = config.get('model', {})
        print(f"\n🤖 模型配置:")
        print(f"  hidden_size: {model_config.get('hidden_size', '未设置')}")
        print(f"  device_target: {model_config.get('device_target', '未设置')}")
        
        # 分析可能导致训练异常的配置
        print(f"\n🔍 训练异常原因分析:")
        
        num_epochs = training_config.get('num_epochs', 5)
        patience = training_config.get('early_stopping_patience', 3)
        
        print(f"  配置的总epoch数: {num_epochs}")
        print(f"  早停耐心值: {patience}")
        
        if num_epochs == 5 and patience == 3:
            print(f"\n⚠️  潜在问题:")
            print(f"  1. 如果WER从epoch 1开始就是0并保持不变")
            print(f"  2. 训练可能会因为没有改善而早停")
            print(f"  3. 或者best model选择逻辑可能有问题")
        
        # 检查是否有关于模型保存的配置
        save_interval = training_config.get('save_interval', 1)
        eval_interval = training_config.get('eval_interval', 1)
        
        print(f"\n💾 模型保存配置:")
        print(f"  保存间隔: 每 {save_interval} epoch")
        print(f"  评估间隔: 每 {eval_interval} epoch")
        
        if save_interval == 1 and eval_interval == 1:
            print(f"  ✓ 应该每个epoch都保存和评估")
            print(f"  ❓ 但实际只保留了epoch 1的权重，说明best model逻辑有问题")
    
    except Exception as e:
        print(f"❌ 无法读取配置文件: {e}")

def main():
    """主函数"""
    
    config_path = "/data/shengteng/training/configs/safe_gpu_config.json"
    
    print("=== 训练异常诊断工具 ===\n")
    
    # 首先读取配置文件获取实际路径
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', "/data/shengteng/training/checkpoints_gpu")
        log_dir = config.get('paths', {}).get('log_dir', "/data/shengteng/training/logs_gpu")
        output_dir = config.get('paths', {}).get('output_dir', "/data/shengteng/training/output_gpu")
        
        print(f"📂 配置文件指定的路径:")
        print(f"  检查点目录: {checkpoint_dir}")
        print(f"  日志目录: {log_dir}")
        print(f"  输出目录: {output_dir}")
        
    except Exception as e:
        print(f"❌ 无法读取配置文件: {e}")
        # 使用默认路径
        checkpoint_dir = "/data/shengteng/training/checkpoints_gpu"
        log_dir = "/data/shengteng/training/logs_gpu"
        output_dir = "/data/shengteng/training/output_gpu"
    
    # 1. 分析训练日志
    print("\n1️⃣ 检查训练日志...")
    analyze_training_logs(log_dir)
    if log_dir != output_dir:
        print("  同时检查输出目录...")
        analyze_training_logs(output_dir)
    
    # 2. 分析模型检查点  
    print("\n2️⃣ 检查模型检查点...")
    analyze_model_checkpoints(checkpoint_dir)
    if checkpoint_dir != output_dir:
        print("  同时检查输出目录...")
        analyze_model_checkpoints(output_dir)
    
    # 3. 分析配置文件
    print("\n3️⃣ 检查配置文件...")
    analyze_config_file(config_path)
    
    print(f"\n=== 诊断总结 ===")
    print(f"🔍 根据用户反馈和分析结果:")
    print(f"  1. WER在训练过程中保持为0")
    print(f"  2. 这导致best model检查点从未更新")
    print(f"  3. 训练实际上只保留了epoch 1的权重")
    print(f"  4. 词汇表包含大量重复的句号标记")
    
    print(f"\n💡 建议的修复方案:")
    print(f"  1. 使用清理后的词汇表 (已生成)")
    print(f"  2. 修改训练配置，使用loss作为监控指标")
    print(f"  3. 设置save_best_only=False以保存所有epoch")
    print(f"  4. 重新开始完整的训练过程")

if __name__ == "__main__":
    main()
