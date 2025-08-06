#!/usr/bin/env python3
"""
SignAvatar Web 系统状态检查和启动脚本
检查系统依赖、配置文件、模型文件等，确保系统正常运行
"""

import os
import sys
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from backend.utils.config import Settings
    from backend.utils.logger import setup_logger
    from backend.utils.database import DatabaseManager
    from backend.utils.cache import CacheManager
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入配置模块: {e}")
    CONFIG_AVAILABLE = False

class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        self.start_time = time.time()
        
        if CONFIG_AVAILABLE:
            self.settings = Settings()
            self.logger = setup_logger("health_checker")
        else:
            self.settings = None
            self.logger = None
    
    def print_header(self):
        """打印检查开始信息"""
        print("=" * 70)
        print("🚀 SignAvatar Web 系统健康检查")
        print("=" * 70)
        print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"项目路径: {project_root}")
        print()
    
    def print_section(self, title: str):
        """打印检查部分标题"""
        print(f"\n📋 {title}")
        print("-" * 50)
    
    def check_item(self, name: str, condition: bool, error_msg: str = "", warning_msg: str = "") -> bool:
        """检查单个项目"""
        if condition:
            print(f"✅ {name}")
            self.checks_passed += 1
            return True
        else:
            if error_msg:
                print(f"❌ {name} - {error_msg}")
                self.errors.append(f"{name}: {error_msg}")
                self.checks_failed += 1
            elif warning_msg:
                print(f"⚠️ {name} - {warning_msg}")
                self.warnings.append(f"{name}: {warning_msg}")
            else:
                print(f"❌ {name}")
                self.checks_failed += 1
            return False
    
    def check_python_environment(self) -> bool:
        """检查Python环境"""
        self.print_section("Python 环境检查")
        
        # Python版本检查
        python_version = sys.version_info
        version_ok = python_version >= (3, 8)
        self.check_item(
            f"Python 版本 ({python_version.major}.{python_version.minor}.{python_version.micro})",
            version_ok,
            "需要 Python 3.8 或更高版本"
        )
        
        # 检查必需的Python包
        required_packages = [
            'fastapi', 'uvicorn', 'websockets', 'pydantic', 'numpy',
            'opencv-python', 'mediapipe', 'aiofiles', 'python-multipart'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.check_item(f"Python包: {package}", True)
            except ImportError:
                self.check_item(f"Python包: {package}", False, "包未安装")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n💡 安装缺失的包: pip install {' '.join(missing_packages)}")
        
        return len(missing_packages) == 0
    
    def check_project_structure(self) -> bool:
        """检查项目结构"""
        self.print_section("项目结构检查")
        
        required_dirs = [
            "backend", "frontend", "models", "data", "logs", "uploads", "temp"
        ]
        
        required_files = [
            "backend/main.py", "backend/utils/config.py", "backend/utils/logger.py",
            "frontend/index.html", "frontend/package.json", "requirements.txt"
        ]
        
        all_good = True
        
        # 检查目录
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            self.check_item(f"目录: {dir_name}/", exists)
            if not exists:
                all_good = False
                # 创建缺失的目录
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"  📁 已创建目录: {dir_name}/")
                except Exception as e:
                    print(f"  ❌ 创建目录失败: {e}")
        
        # 检查文件
        for file_name in required_files:
            file_path = project_root / file_name
            exists = file_path.exists() and file_path.is_file()
            self.check_item(f"文件: {file_name}", exists)
            if not exists:
                all_good = False
        
        return all_good
    
    def check_configuration(self) -> bool:
        """检查配置文件"""
        self.print_section("配置文件检查")
        
        if not CONFIG_AVAILABLE:
            self.check_item("配置模块", False, "无法导入配置模块")
            return False
        
        # 检查 .env 文件
        env_file = project_root / ".env"
        env_exists = env_file.exists()
        self.check_item(".env 配置文件", env_exists)
        
        if not env_exists:
            # 创建默认配置文件
            self.create_default_env_file()
        
        # 检查配置有效性
        try:
            config_warnings = self.settings.validate_environment()
            if config_warnings:
                for warning in config_warnings:
                    self.warnings.append(f"配置警告: {warning}")
                    print(f"⚠️ 配置警告: {warning}")
            
            self.check_item("配置验证", len(config_warnings) == 0)
            return True
            
        except Exception as e:
            self.check_item("配置验证", False, str(e))
            return False
    
    def check_model_files(self) -> bool:
        """检查AI模型文件"""
        self.print_section("AI 模型文件检查")
        
        if not CONFIG_AVAILABLE:
            return False
        
        model_files = [
            (self.settings.CSLR_MODEL_PATH, "CSLR模型"),
            (self.settings.DIFFUSION_MODEL_PATH, "Diffusion模型"),
            (self.settings.FEDERATED_MODEL_PATH, "联邦学习模型"),
            (self.settings.CSLR_VOCAB_PATH, "词汇表文件"),
        ]
        
        all_good = True
        for model_path, model_name in model_files:
            file_path = Path(model_path)
            exists = file_path.exists()
            
            if exists:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                self.check_item(f"{model_name} ({size_mb:.1f}MB)", True)
            else:
                self.check_item(
                    model_name, 
                    False, 
                    f"文件不存在: {model_path}",
                    "模型文件缺失，将使用模拟模式"
                )
                all_good = False
        
        if not all_good:
            print("\n💡 提示: 模型文件缺失时系统将使用模拟模式运行")
            print("   请联系项目维护者获取完整的模型文件")
        
        return True  # 即使模型文件缺失也允许继续运行
    
    def check_database(self) -> bool:
        """检查数据库连接"""
        self.print_section("数据库检查")
        
        if not CONFIG_AVAILABLE:
            return False
        
        try:
            # 创建数据库管理器实例
            db_manager = DatabaseManager()
            
            # 测试数据库连接
            import asyncio
            async def test_db():
                try:
                    await db_manager.initialize()
                    await db_manager.cleanup()
                    return True
                except Exception as e:
                    print(f"数据库连接失败: {e}")
                    return False
            
            # 运行异步测试
            db_ok = asyncio.run(test_db())
            self.check_item("数据库连接", db_ok)
            
            # 检查数据库文件权限
            db_dir = Path(self.settings.DATABASE_URL.replace("sqlite:///", "")).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            writable = os.access(db_dir, os.W_OK)
            self.check_item("数据库目录写权限", writable)
            
            return db_ok and writable
            
        except Exception as e:
            self.check_item("数据库检查", False, str(e))
            return False
    
    def check_network_ports(self) -> bool:
        """检查网络端口"""
        self.print_section("网络端口检查")
        
        if not CONFIG_AVAILABLE:
            return False
        
        import socket
        
        def check_port(port: int, name: str) -> bool:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    return result != 0  # 端口可用（未被占用）
            except Exception:
                return True  # 假设可用
        
        ports_to_check = [
            (self.settings.PORT, "主服务端口"),
            (self.settings.METRICS_PORT, "监控端口"),
        ]
        
        all_good = True
        for port, name in ports_to_check:
            available = check_port(port, name)
            self.check_item(f"{name} ({port})", available, f"端口 {port} 已被占用")
            if not available:
                all_good = False
        
        return all_good
    
    def check_frontend_dependencies(self) -> bool:
        """检查前端依赖"""
        self.print_section("前端依赖检查")
        
        frontend_dir = project_root / "frontend"
        if not frontend_dir.exists():
            self.check_item("前端目录", False, "frontend目录不存在")
            return False
        
        # 检查 package.json
        package_json = frontend_dir / "package.json"
        package_exists = package_json.exists()
        self.check_item("package.json", package_exists)
        
        # 检查 node_modules
        node_modules = frontend_dir / "node_modules"
        modules_exists = node_modules.exists()
        self.check_item("node_modules", modules_exists, "请运行 'npm install' 安装依赖")
        
        # 检查关键文件
        key_files = ["index.html", "src/main.tsx", "src/App.tsx"]
        for file_name in key_files:
            file_path = frontend_dir / file_name
            exists = file_path.exists()
            self.check_item(f"前端文件: {file_name}", exists)
        
        return package_exists and modules_exists
    
    def create_default_env_file(self):
        """创建默认的.env配置文件"""
        env_content = """# SignAvatar Web 环境配置文件

# 基本设置
APP_NAME=SignAvatar Web
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# 服务器设置
HOST=0.0.0.0
PORT=8000

# 安全设置（生产环境请修改）
SECRET_KEY=your-secret-key-here-change-in-production

# 数据库设置
DATABASE_URL=sqlite:///./data/signavatar.db

# Redis设置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI模型设置
CSLR_MODEL_PATH=models/cslr_transformer.mindir
DIFFUSION_MODEL_PATH=models/diffusion_slp.mindir
FEDERATED_MODEL_PATH=models/federated_slr.mindir

# MediaPipe设置
MEDIAPIPE_MODEL_COMPLEXITY=1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE=0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE=0.5

# 性能设置
MAX_WEBSOCKET_CONNECTIONS=100
FRAME_BUFFER_SIZE=30
INFERENCE_BATCH_SIZE=1

# 日志设置
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090

# 文件上传设置
MAX_UPLOAD_SIZE=100
UPLOAD_DIR=uploads
TEMP_DIR=temp
"""
        
        try:
            env_file = project_root / ".env"
            env_file.write_text(env_content)
            print("📝 已创建默认 .env 配置文件")
            return True
        except Exception as e:
            print(f"❌ 创建 .env 文件失败: {e}")
            return False
    
    def run_comprehensive_check(self) -> Dict:
        """运行全面的系统检查"""
        self.print_header()
        
        checks = [
            ("Python环境", self.check_python_environment),
            ("项目结构", self.check_project_structure),
            ("配置文件", self.check_configuration),
            ("AI模型文件", self.check_model_files),
            ("数据库", self.check_database),
            ("网络端口", self.check_network_ports),
            ("前端依赖", self.check_frontend_dependencies),
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                results[check_name] = check_func()
            except Exception as e:
                print(f"❌ {check_name}检查时发生错误: {e}")
                results[check_name] = False
                self.errors.append(f"{check_name}: {e}")
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results: Dict):
        """打印检查总结"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("📊 检查总结")
        print("=" * 70)
        
        print(f"✅ 通过检查: {self.checks_passed}")
        print(f"❌ 失败检查: {self.checks_failed}")
        print(f"⚠️ 警告数量: {len(self.warnings)}")
        print(f"⏱️ 检查耗时: {elapsed_time:.2f}秒")
        
        # 显示警告
        if self.warnings:
            print(f"\n⚠️ 警告信息:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # 显示错误
        if self.errors:
            print(f"\n❌ 错误信息:")
            for error in self.errors:
                print(f"   • {error}")
        
        # 整体状态
        print("\n" + "=" * 70)
        if self.checks_failed == 0:
            print("🎉 系统健康检查全部通过！可以启动服务")
            print("\n🚀 启动命令:")
            print("   后端: python start.py start")
            print("   前端: cd frontend && npm run dev")
        elif self.checks_failed <= 2 and len(self.errors) == 0:
            print("⚠️ 系统基本正常，有少量警告，可以启动服务")
            print("   建议解决警告后再启动")
        else:
            print("❌ 系统存在重要问题，建议解决后再启动服务")
            print("\n🔧 建议的修复步骤:")
            print("   1. 安装缺失的Python依赖: pip install -r requirements.txt")
            print("   2. 安装前端依赖: cd frontend && npm install")
            print("   3. 检查配置文件 .env 的设置")
            print("   4. 确保必要的目录存在且有写权限")
        
        print("=" * 70)


def main():
    """主函数"""
    checker = SystemHealthChecker()
    results = checker.run_comprehensive_check()
    
    # 返回适当的退出码
    if checker.checks_failed == 0:
        return 0  # 成功
    elif checker.checks_failed <= 2:
        return 1  # 警告
    else:
        return 2  # 错误


if __name__ == "__main__":
    sys.exit(main())