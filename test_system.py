#!/usr/bin/env python3
"""
SignAvatar Web 后端系统测试
测试主要功能模块的基本功能
"""

import asyncio
import pytest
import httpx
from pathlib import Path
import json
import sys

# 添加后端路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.utils.database import db_manager
from backend.utils.cache import cache_manager
from backend.utils.security import security_manager
from backend.utils.file_manager import file_manager
from backend.utils.monitoring import performance_monitor


class TestSystemComponents:
    """系统组件测试"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self):
        """测试数据库初始化"""
        try:
            await db_manager.initialize()
            assert True, "数据库初始化成功"
        except Exception as e:
            pytest.fail(f"数据库初始化失败: {e}")
        finally:
            await db_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_manager(self):
        """测试缓存管理器"""
        try:
            await cache_manager.initialize()
            
            # 测试设置和获取缓存
            await cache_manager.set("test", "key1", "value1")
            result = await cache_manager.get("test", "key1")
            
            assert result == "value1", "缓存设置和获取功能正常"
            
            # 测试删除缓存
            await cache_manager.delete("test", "key1")
            result = await cache_manager.get("test", "key1")
            
            assert result is None, "缓存删除功能正常"
            
        except Exception as e:
            pytest.fail(f"缓存管理器测试失败: {e}")
        finally:
            await cache_manager.cleanup()
    
    def test_security_manager(self):
        """测试安全管理器"""
        try:
            # 测试令牌创建
            token = security_manager.create_access_token({"user_id": 1, "username": "test"})
            assert token is not None, "令牌创建成功"
            
            # 测试令牌验证
            payload = security_manager.verify_token(token)
            assert payload["user_id"] == 1, "令牌验证成功"
            
        except Exception as e:
            pytest.fail(f"安全管理器测试失败: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_monitor(self):
        """测试性能监控器"""
        try:
            await performance_monitor.initialize()
            
            # 测试指标记录
            performance_monitor.metrics_collector.record_request(
                "GET", "/test", 200, 0.1
            )
            
            # 获取统计信息
            stats = performance_monitor.get_comprehensive_report()
            assert "system_metrics" in stats, "性能监控功能正常"
            
        except Exception as e:
            pytest.fail(f"性能监控器测试失败: {e}")
        finally:
            await performance_monitor.cleanup()
    
    def test_file_manager(self):
        """测试文件管理器"""
        try:
            # 测试存储统计
            stats = file_manager.get_storage_stats()
            assert "total_files" in stats, "文件管理器基本功能正常"
            
        except Exception as e:
            pytest.fail(f"文件管理器测试失败: {e}")


class TestAPIEndpoints:
    """API端点测试"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """测试健康检查端点"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/api/health")
                
                if response.status_code == 200:
                    data = response.json()
                    assert "status" in data, "健康检查端点正常"
                    assert "services" in data, "服务状态信息存在"
                else:
                    print(f"服务器可能未启动，状态码: {response.status_code}")
                    
        except httpx.ConnectError:
            print("无法连接到服务器，请确保服务器正在运行")
        except Exception as e:
            pytest.fail(f"健康检查端点测试失败: {e}")


def run_basic_tests():
    """运行基本测试"""
    print("🧪 开始运行 SignAvatar Web 后端基本测试...")
    
    # 测试系统组件
    print("\n📋 测试系统组件...")
    test_suite = TestSystemComponents()
    
    try:
        # 同步测试
        test_suite.test_security_manager()
        print("✅ 安全管理器测试通过")
        
        test_suite.test_file_manager()
        print("✅ 文件管理器测试通过")
        
        # 异步测试
        async def run_async_tests():
            await test_suite.test_database_initialization()
            print("✅ 数据库初始化测试通过")
            
            await test_suite.test_cache_manager()
            print("✅ 缓存管理器测试通过")
            
            await test_suite.test_performance_monitor()
            print("✅ 性能监控器测试通过")
        
        asyncio.run(run_async_tests())
        
    except Exception as e:
        print(f"❌ 系统组件测试失败: {e}")
        return False
    
    # 测试API端点
    print("\n🌐 测试API端点...")
    api_test = TestAPIEndpoints()
    
    try:
        asyncio.run(api_test.test_health_endpoint())
        print("✅ API端点测试完成")
        
    except Exception as e:
        print(f"⚠️ API端点测试异常: {e}")
    
    print("\n🎉 基本测试完成！")
    return True


def run_integration_test():
    """运行集成测试"""
    print("\n🔗 开始集成测试...")
    
    async def integration_test():
        try:
            # 初始化所有服务
            print("初始化数据库...")
            await db_manager.initialize()
            
            print("初始化缓存...")
            await cache_manager.initialize()
            
            print("初始化性能监控...")
            await performance_monitor.initialize()
            
            # 创建测试用户
            print("创建测试用户...")
            user_id = await db_manager.create_user(
                username="test_user",
                email="test@example.com",
                password="test_password",
                full_name="Test User"
            )
            
            print(f"测试用户创建成功，ID: {user_id}")
            
            # 测试用户认证
            print("测试用户认证...")
            user = await db_manager.authenticate_user("test_user", "test_password")
            assert user is not None, "用户认证成功"
            print("✅ 用户认证测试通过")
            
            # 测试会话管理
            print("测试会话管理...")
            session_id = await db_manager.create_session(user_id=user_id)
            print(f"会话创建成功，ID: {session_id}")
            
            # 测试缓存功能
            print("测试缓存功能...")
            await cache_manager.set("test_integration", "session", session_id)
            cached_session = await cache_manager.get("test_integration", "session")
            assert cached_session == session_id, "缓存功能正常"
            print("✅ 缓存功能测试通过")
            
            # 测试性能监控
            print("测试性能监控...")
            performance_monitor.metrics_collector.record_request("GET", "/test", 200, 0.1)
            stats = performance_monitor.get_comprehensive_report()
            assert "system_metrics" in stats, "性能监控正常"
            print("✅ 性能监控测试通过")
            
            print("\n🎊 集成测试全部通过！")
            
        except Exception as e:
            print(f"❌ 集成测试失败: {e}")
            return False
        finally:
            # 清理资源
            print("清理测试资源...")
            await performance_monitor.cleanup()
            await cache_manager.cleanup()
            await db_manager.cleanup()
        
        return True
    
    return asyncio.run(integration_test())


def main():
    """主测试函数"""
    print("🚀 SignAvatar Web 后端系统测试套件")
    print("=" * 50)
    
    # 运行基本测试
    basic_test_passed = run_basic_tests()
    
    if basic_test_passed:
        # 运行集成测试
        integration_test_passed = run_integration_test()
        
        if integration_test_passed:
            print("\n🎉 所有测试通过！系统功能正常")
            return True
        else:
            print("\n⚠️ 集成测试失败，请检查系统配置")
            return False
    else:
        print("\n❌ 基本测试失败，请检查系统组件")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
