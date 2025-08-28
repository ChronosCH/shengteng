#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语学习训练系统启动器
专注于手语教学功能，而非模型训练
"""

import os
import sys
import subprocess
from pathlib import Path
import webbrowser
import time

def main():
    """主函数"""
    print("� 手语学习训练系统")
    print("=" * 50)
    print("系统化学习手语，掌握沟通技能")
    print()
    
    print("📚 可用的学习选项:")
    print("1. 🌐 启动Web学习平台 - 完整的在线学习体验")
    print("2. 📱 本地学习演示 - 快速体验学习功能")
    print("3. � 查看学习统计 - 分析学习进度和成果")
    print("4. ⚙️ 系统配置 - 配置学习参数")
    print("5. ℹ️ 查看帮助信息")
    print()
    
    while True:
        choice = input("请选择学习选项 (1-5, q退出): ").strip()
        
        if choice.lower() == 'q':
            print("👋 感谢使用手语学习训练系统！")
            break
        elif choice == '1':
            print("🚀 启动Web学习平台...")
            start_web_platform()
            break
        elif choice == '2':
            print("� 启动本地学习演示...")
            start_local_demo()
            break
        elif choice == '3':
            print("📊 显示学习统计...")
            show_learning_stats()
        elif choice == '4':
            print("⚙️ 打开系统配置...")
            show_config()
        elif choice == '5':
            show_help()
        else:
            print("❌ 无效选择，请输入 1-5 或 q")

def start_web_platform():
    """启动Web学习平台"""
    try:
        print("\n🌐 正在启动Web学习平台...")
        print("📍 平台地址: http://localhost:5173")
        print("🎯 功能特色:")
        print("  • 系统化学习路径")
        print("  • 互动式手语练习")
        print("  • 实时进度跟踪")
        print("  • 成就系统激励")
        print("  • 社交学习功能")
        print()
        
        # 检查前端目录
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("❌ 错误: frontend目录不存在")
            return
        
        # 启动前端开发服务器
        print("正在启动前端服务...")
        os.chdir(str(frontend_dir))
        
        # 检查是否安装了依赖
        if not Path("node_modules").exists():
            print("📦 正在安装依赖...")
            subprocess.run(["npm", "install"], check=True)
        
        # 启动开发服务器
        print("🎬 启动开发服务器...")
        time.sleep(2)
        
        # 在浏览器中打开
        webbrowser.open("http://localhost:5173/learning")
        
        # 启动开发服务器（这会阻塞）
        subprocess.run(["npm", "run", "dev"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已安装 Node.js 和 npm")
    except FileNotFoundError:
        print("❌ 错误: 无法找到 npm 命令")
        print("💡 请先安装 Node.js 环境")
    except KeyboardInterrupt:
        print("\n⚠️ 服务被用户中断")
    finally:
        # 返回原目录
        os.chdir("..")

def start_local_demo():
    """启动本地学习演示"""
    try:
        print("\n📱 启动本地学习演示...")
        print("🎯 演示功能:")
        print("  • 基础手语词汇展示")
        print("  • 数字手语练习")
        print("  • 简单交互测试")
        print()
        
        # 检查后端目录
        backend_dir = Path("backend")
        if not backend_dir.exists():
            print("❌ 错误: backend目录不存在")
            return
        
        # 启动简化的学习演示
        demo_script = backend_dir / "demo_learning.py"
        if demo_script.exists():
            subprocess.run([sys.executable, str(demo_script)], check=True)
        else:
            print("🎓 学习演示功能")
            print("-" * 30)
            print("1. 基础手语词汇:")
            print("   • 你好 👋")
            print("   • 谢谢 🙏") 
            print("   • 再见 👋")
            print()
            print("2. 数字手语 (0-10):")
            print("   • 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣")
            print("   • 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟")
            print()
            print("3. 家庭称谓:")
            print("   • 爸爸 👨 妈妈 👩")
            print("   • 哥哥 👦 姐姐 👧")
            print()
            print("✨ 完整学习体验请选择选项1启动Web平台")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 演示启动失败: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")

def show_learning_stats():
    """显示学习统计"""
    print("\n📊 学习统计概览")
    print("-" * 30)
    print()
    
    # 模拟统计数据
    stats = {
        "总学习时间": "245分钟",
        "完成课程": "28节",
        "当前等级": "15级",
        "连续学习": "7天",
        "掌握词汇": "156个",
        "获得成就": "12个"
    }
    
    for key, value in stats.items():
        print(f"📈 {key}: {value}")
    
    print()
    print("🎯 本周目标:")
    print("  • 学习时间: 180/300分钟 (60%)")
    print("  • 完成课程: 12/15节 (80%)")
    print("  • 练习次数: 25/30次 (83%)")
    print()
    
    print("🏆 最近成就:")
    print("  • ✅ 坚持一周 (连续学习7天)")
    print("  • ✅ 基础入门 (完成基础课程)")
    print("  • 🔄 学习达人 (进度: 3/5)")
    print()

def show_config():
    """显示系统配置"""
    print("\n⚙️ 系统配置")
    print("-" * 30)
    print()
    print("📚 学习配置:")
    print("  • 每日学习目标: 30分钟")
    print("  • 难度设置: 自适应")
    print("  • 提醒设置: 开启")
    print("  • 语音反馈: 开启")
    print()
    print("🎨 界面设置:")
    print("  • 主题: 马卡龙色彩")
    print("  • 语言: 中文")
    print("  • 动画效果: 开启")
    print()
    print("📊 数据设置:")
    print("  • 学习记录: 自动保存")
    print("  • 进度同步: 启用")
    print("  • 隐私保护: 启用")
    print()
    print("💡 如需修改配置，请在Web平台的设置页面进行调整")
    print()

def show_help():
    """显示帮助信息"""
    print()
    print("📖 手语学习训练系统帮助")
    print("-" * 40)
    print()
    print("🎯 系统特色:")
    print("• 系统化学习路径  : 从基础到高级的完整学习体系")
    print("• 互动式练习     : 实时手语识别和反馈")
    print("• 进度跟踪       : 详细的学习进度和成就系统") 
    print("• 个性化推荐     : 基于学习情况的智能推荐")
    print("• 社交学习       : 与其他学习者互动交流")
    print()
    print("📚 学习内容:")
    print("• 基础手语       : 问候语、自我介绍、常用词汇")
    print("• 数字时间       : 数字表达、时间概念")
    print("• 家庭关系       : 家庭成员、人际关系")
    print("• 日常活动       : 生活场景、动作表达")
    print("• 高级语法       : 复杂语法、专业表达")
    print()
    print("🛠️ 系统要求:")
    print("• Node.js 16+    : 前端开发环境")
    print("• Python 3.8+   : 后端运行环境")
    print("• 现代浏览器     : Chrome, Firefox, Safari等")
    print("• 摄像头权限     : 用于手语识别功能")
    print()
    print("🆘 获取帮助:")
    print("• 在线文档       : docs/user-guide.md")
    print("• 问题反馈       : 通过Web平台反馈功能")
    print("• 社区讨论       : 加入学习交流群")
    print()
    print("🔗 相关链接:")
    print("• Web平台        : http://localhost:5173")
    print("• 学习页面       : http://localhost:5173/learning")
    print("• API文档        : http://localhost:8000/docs")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序退出")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        print("💡 请检查系统环境或联系技术支持")
