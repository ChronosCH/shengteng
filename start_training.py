#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE-CSL手语识别系统 - 全局训练启动器
优化后的架构，简化的入口
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("🎯 CE-CSL手语识别训练系统启动器")
    print("=" * 50)
    print()
    
    # 检查training目录
    training_dir = Path("training")
    if not training_dir.exists():
        print("❌ 错误: training目录不存在")
        return
    
    print("📁 可用的训练选项:")
    print("1. 🏆 最优训练器 (optimal) - 45% 准确率")
    print("2. 🔧 增强训练器 (enhanced) - 37.5% 准确率")
    print("3. ℹ️ 查看帮助信息")
    print("4. 📊 查看详细技术报告")
    print()
    
    while True:
        choice = input("请选择训练选项 (1-4, q退出): ").strip()
        
        if choice.lower() == 'q':
            print("👋 再见！")
            break
        elif choice == '1':
            print("🚀 启动最优训练器...")
            run_training('optimal')
            break
        elif choice == '2':
            print("🔧 启动增强训练器...")
            run_training('enhanced')
            break
        elif choice == '3':
            show_help()
        elif choice == '4':
            show_report()
        else:
            print("❌ 无效选择，请输入 1-4 或 q")

def run_training(model_type):
    """运行训练"""
    try:
        # 切换到training目录
        os.chdir("training")
        
        # 构建命令
        cmd = [sys.executable, "train.py", "--model", model_type]
        
        print(f"执行命令: {' '.join(cmd)}")
        print()
        
        # 运行训练
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("🎉 训练成功完成！")
        else:
            print("❌ 训练过程中出现错误")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
    except FileNotFoundError:
        print("❌ 错误: 无法找到train.py文件")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")

def show_help():
    """显示帮助信息"""
    print()
    print("📖 帮助信息")
    print("-" * 30)
    print()
    print("训练器对比:")
    print("• optimal_trainer  : 45% 准确率，注意力机制，Focal Loss")
    print("• enhanced_trainer : 37.5% 准确率，数据增强，稳定训练")
    print()
    print("手动训练命令:")
    print("cd training")
    print("python train.py --model optimal         # 最优训练器")
    print("python train.py --model enhanced        # 增强训练器")
    print("python train.py --help                  # 更多参数")
    print()
    print("输出位置:")
    print("• 模型文件: checkpoints/")
    print("• 训练日志: training/training.log")
    print("• 技术报告: training/最优训练总结报告.md")
    print()

def show_report():
    """显示技术报告"""
    report_path = Path("training/最优训练总结报告.md")
    if report_path.exists():
        print()
        print("📊 技术报告已生成")
        print(f"📁 位置: {report_path.absolute()}")
        print()
        print("主要成果:")
        print("• 最佳准确率: 45.0%")
        print("• 性能提升: 125% (从20%到45%)")
        print("• 创新技术: 6项突破性技术")
        print("• 训练架构: 完全优化")
        print()
        
        try:
            # 在默认程序中打开报告
            if sys.platform.startswith('win'):
                os.startfile(str(report_path))
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', str(report_path)])
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(report_path)])
            print("📖 技术报告已在默认程序中打开")
        except:
            print(f"💡 请手动打开文件: {report_path}")
    else:
        print("❌ 技术报告文件不存在")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序退出")
