"""
简化的系统验证脚本
快速检查核心功能是否正常
"""

import sys
import os
from pathlib import Path

def check_project_structure():
    """检查项目结构"""
    print("🔍 检查项目结构...")
    
    required_dirs = [
        "backend",
        "backend/services", 
        "backend/utils",
        "backend/api",
        "frontend",
        "models"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False
    
    print("✅ 项目结构检查通过")
    return True


def check_required_files():
    """检查必需文件"""
    print("\n📁 检查必需文件...")
    
    required_files = [
        "backend/main.py",
        "backend/utils/config.py",
        "backend/utils/logger.py",
        "backend/services/mediapipe_service.py",
        "requirements.txt",
        "start.py",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    print("✅ 必需文件检查通过")
    return True


def check_python_imports():
    """检查Python模块导入"""
    print("\n🐍 检查Python模块...")
    
    basic_modules = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy"
    ]
    
    failed_imports = []
    for module in basic_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            failed_imports.append(module)
            print(f"❌ {module} (未安装)")
    
    if failed_imports:
        print(f"⚠️ 需要安装: pip install {' '.join(failed_imports)}")
        return False
    
    print("✅ 基础模块检查通过")
    return True


def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 检查配置文件...")
    
    # 检查环境配置
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists():
        print("✅ .env.example 存在")
        if not env_file.exists():
            print("ℹ️ 可以复制 .env.example 到 .env")
    else:
        print("❌ .env.example 不存在")
    
    # 检查模型目录
    models_dir = Path("models")
    if models_dir.exists():
        print("✅ models 目录存在")
        model_files = list(models_dir.glob("*.json"))
        if model_files:
            print(f"✅ 找到 {len(model_files)} 个配置文件")
        else:
            print("ℹ️ models 目录为空，需要添加模型文件")
    else:
        print("❌ models 目录不存在")
    
    return True


def test_basic_functionality():
    """测试基础功能"""
    print("\n🧪 测试基础功能...")
    
    try:
        # 尝试导入主要服务类
        sys.path.insert(0, str(Path(__file__).parent))
        
        # 测试配置加载
        try:
            from backend.utils.config import settings
            print("✅ 配置模块加载成功")
            print(f"   - 应用名称: {settings.APP_NAME}")
            print(f"   - 版本: {settings.VERSION}")
            print(f"   - 调试模式: {settings.DEBUG}")
        except Exception as e:
            print(f"❌ 配置模块加载失败: {e}")
            return False
        
        # 测试日志模块
        try:
            from backend.utils.logger import setup_logger
            logger = setup_logger("test")
            logger.info("测试日志消息")
            print("✅ 日志模块工作正常")
        except Exception as e:
            print(f"❌ 日志模块测试失败: {e}")
            return False
        
        print("✅ 基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False


def create_missing_directories():
    """创建缺失的目录"""
    print("\n📂 创建必需目录...")
    
    dirs_to_create = [
        "data",
        "logs", 
        "uploads",
        "uploads/image",
        "uploads/video",
        "uploads/audio",
        "uploads/document",
        "uploads/data",
        "temp",
        "models"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print("✅ 目录创建完成")


def generate_env_file():
    """生成环境配置文件"""
    print("\n⚙️ 生成环境配置...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ .env 文件已创建")
    elif env_file.exists():
        print("✅ .env 文件已存在")
    else:
        print("⚠️ 无法创建 .env 文件")


def main():
    """主函数"""
    print("🚀 SignAvatar Web 系统验证")
    print("=" * 50)
    
    all_checks_passed = True
    
    # 运行各项检查
    checks = [
        check_project_structure,
        check_required_files,
        check_python_imports,
        check_config_files,
        test_basic_functionality
    ]
    
    for check in checks:
        if not check():
            all_checks_passed = False
    
    print("\n" + "=" * 50)
    
    if all_checks_passed:
        print("🎉 所有检查通过！系统已准备就绪")
        print("\n📝 下一步操作:")
        print("1. 确认 .env 文件配置正确")
        print("2. 添加必要的AI模型文件到 models/ 目录")
        print("3. 运行 'python start.py start' 启动开发服务器")
        print("4. 访问 http://localhost:8000/api/docs 查看API文档")
        return True
    else:
        print("⚠️ 部分检查未通过，建议先解决这些问题")
        print("\n🔧 自动修复:")
        create_missing_directories()
        generate_env_file()
        print("\n再次运行此脚本检查问题是否解决")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)