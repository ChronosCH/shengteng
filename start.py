#!/usr/bin/env python3
"""
SignAvatar Web 后端启动脚本
提供开发和生产环境的启动选项
"""

import asyncio
import os
import sys
import click
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.utils.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@click.group()
def cli():
    """SignAvatar Web 后端管理工具"""
    pass


@cli.command()
@click.option('--host', default=settings.HOST, help='服务器主机地址')
@click.option('--port', default=settings.PORT, help='服务器端口')
@click.option('--reload', is_flag=True, help='启用热重载（开发模式）')
@click.option('--workers', default=1, help='工作进程数量')
@click.option('--log-level', default='info', help='日志级别')
def start(host, port, reload, workers, log_level):
    """启动后端服务器"""
    click.echo(f"🚀 启动 SignAvatar Web 后端服务器...")
    click.echo(f"📍 地址: http://{host}:{port}")
    click.echo(f"📊 API文档: http://{host}:{port}/api/docs")
    
    # 确保必要的目录存在
    ensure_directories()
    
    # 启动服务器
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
        access_log=True
    )


@cli.command()
def init():
    """初始化项目环境"""
    click.echo("🔧 初始化 SignAvatar Web 后端环境...")
    
    # 创建必要的目录
    ensure_directories()
    
    # 复制环境配置文件
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        click.echo("✅ 创建了 .env 配置文件")
    
    # 创建默认管理员用户（如果不存在）
    click.echo("👤 设置管理员账户...")
    
    click.echo("✨ 环境初始化完成！")
    click.echo("\n📝 接下来的步骤:")
    click.echo("1. 编辑 .env 文件配置你的环境参数")
    click.echo("2. 运行 'python start.py start' 启动服务器")
    click.echo("3. 访问 http://localhost:8000/api/docs 查看API文档")


@cli.command()
@click.option('--test-type', default='all', help='测试类型: unit, integration, all')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def test(test_type, verbose):
    """运行测试套件"""
    import subprocess
    
    click.echo(f"🧪 运行测试套件 ({test_type})...")
    
    test_args = ["python", "-m", "pytest"]
    
    if verbose:
        test_args.append("-v")
    
    if test_type == "unit":
        test_args.append("tests/unit/")
    elif test_type == "integration":
        test_args.append("tests/integration/")
    else:
        test_args.append("tests/")
    
    result = subprocess.run(test_args)
    
    if result.returncode == 0:
        click.echo("✅ 所有测试通过！")
    else:
        click.echo("❌ 部分测试失败")
        sys.exit(1)


@cli.command()
def check_health():
    """检查系统健康状态"""
    import httpx
    import asyncio
    
    async def check():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{settings.HOST}:{settings.PORT}/api/health")
                
                if response.status_code == 200:
                    data = response.json()
                    click.echo("✅ 服务器健康状态良好")
                    click.echo(f"状态: {data.get('status')}")
                    click.echo(f"消息: {data.get('message')}")
                    
                    services = data.get('services', {})
                    click.echo("\n📊 服务状态:")
                    for service, status in services.items():
                        icon = "✅" if status == "ready" else "❌"
                        click.echo(f"  {icon} {service}: {status}")
                else:
                    click.echo(f"❌ 服务器响应异常: {response.status_code}")
        
        except Exception as e:
            click.echo(f"❌ 无法连接到服务器: {e}")
            click.echo("请确保服务器正在运行")
    
    asyncio.run(check())


@cli.command()
@click.option('--output', default='performance_report.json', help='输出文件路径')
def performance_report(output):
    """生成性能报告"""
    import httpx
    import asyncio
    import json
    
    async def generate_report():
        try:
            async with httpx.AsyncClient() as client:
                # 获取系统指标
                response = await client.get(
                    f"http://{settings.HOST}:{settings.PORT}/api/admin/metrics",
                    headers={"Authorization": "Bearer your-admin-token"}  # 需要实际的管理员令牌
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 保存报告
                    with open(output, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    click.echo(f"✅ 性能报告已保存到: {output}")
                else:
                    click.echo(f"❌ 获取性能数据失败: {response.status_code}")
        
        except Exception as e:
            click.echo(f"❌ 生成性能报告失败: {e}")
    
    asyncio.run(generate_report())


@cli.command()
def create_admin():
    """创建管理员用户"""
    import asyncio
    from backend.utils.database import db_manager
    
    async def create():
        try:
            await db_manager.initialize()
            
            username = click.prompt("管理员用户名")
            email = click.prompt("管理员邮箱")
            password = click.prompt("管理员密码", hide_input=True)
            full_name = click.prompt("管理员全名", default="")
            
            # 创建管理员用户
            user_id = await db_manager.create_user(
                username=username,
                email=email,
                password=password,
                full_name=full_name,
                preferences={"role": "admin"},
                accessibility_settings={}
            )
            
            click.echo(f"✅ 管理员用户创建成功，ID: {user_id}")
            
        except Exception as e:
            click.echo(f"❌ 创建管理员用户失败: {e}")
        finally:
            await db_manager.cleanup()
    
    asyncio.run(create())


def ensure_directories():
    """确保必要的目录存在"""
    directories = [
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
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cli()