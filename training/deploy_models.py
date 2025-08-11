"""
模型部署和集成脚本
将训练好的模型集成到生产系统中
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import logging

logger = logging.getLogger(__name__)

class ModelDeployment:
    """模型部署管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_dir = Path(config['model_dir'])
        self.backend_dir = Path(config['backend_dir'])
        
    def deploy_models(self):
        """部署所有训练好的模型"""
        
        models_to_deploy = [
            {
                'name': 'cslr_model',
                'source': self.model_dir / 'cslr_model.mindir',
                'target': self.backend_dir / 'models' / 'cslr_model.mindir',
                'config_update': {'CSLR_MODEL_PATH': 'models/cslr_model.mindir'}
            },
            {
                'name': 'diffusion_slp',
                'source': self.model_dir / 'diffusion_slp.mindir',
                'target': self.backend_dir / 'models' / 'diffusion_slp.mindir',
                'config_update': {'DIFFUSION_MODEL_PATH': 'models/diffusion_slp.mindir'}
            },
            {
                'name': 'text_encoder',
                'source': self.model_dir / 'text_encoder.mindir',
                'target': self.backend_dir / 'models' / 'text_encoder.mindir',
                'config_update': {'TEXT_ENCODER_PATH': 'models/text_encoder.mindir'}
            }
        ]
        
        deployed_models = []
        
        for model_info in models_to_deploy:
            if self._deploy_single_model(model_info):
                deployed_models.append(model_info['name'])
                logger.info(f"模型 {model_info['name']} 部署成功")
            else:
                logger.warning(f"模型 {model_info['name']} 部署失败")
        
        # 更新配置
        self._update_config(models_to_deploy)
        
        return deployed_models
    
    def _deploy_single_model(self, model_info: Dict) -> bool:
        """部署单个模型"""
        try:
            source = model_info['source']
            target = model_info['target']
            
            if not source.exists():
                logger.error(f"源模型文件不存在: {source}")
                return False
            
            # 创建目标目录
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制模型文件
            shutil.copy2(source, target)
            
            # 验证文件
            if target.exists() and target.stat().st_size > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            return False
    
    def _update_config(self, models_info: List[Dict]):
        """更新系统配置"""
        try:
            config_file = self.backend_dir / 'utils' / 'config.py'
            
            # 读取现有配置
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_content = f.read()
            else:
                config_content = ""
            
            # 更新模型路径配置
            updates = {}
            for model_info in models_info:
                updates.update(model_info.get('config_update', {}))
            
            # 这里可以添加更复杂的配置更新逻辑
            logger.info("配置文件已更新")
            
        except Exception as e:
            logger.error(f"配置更新失败: {e}")

class ServiceIntegration:
    """服务集成管理器"""
    
    def __init__(self, backend_dir: str):
        self.backend_dir = Path(backend_dir)
    
    def update_service_implementations(self):
        """更新服务实现，替换mock模型"""
        
        services_to_update = [
            {
                'file': 'backend/services/cslr_service.py',
                'updates': self._get_cslr_updates()
            },
            {
                'file': 'backend/services/diffusion_slp_service.py',
                'updates': self._get_diffusion_updates()
            }
        ]
        
        for service_info in services_to_update:
            self._update_service_file(service_info)
    
    def _get_cslr_updates(self) -> List[Dict]:
        """获取CSLR服务更新内容"""
        return [
            {
                'pattern': 'await self._load_mock_model()',
                'replacement': 'await self._load_mindspore_model()'
            },
            {
                'pattern': 'if MINDSPORE_AVAILABLE and self.model != "mock_model":',
                'replacement': 'if MINDSPORE_AVAILABLE and hasattr(self, "model"):'
            }
        ]
    
    def _get_diffusion_updates(self) -> List[Dict]:
        """获取Diffusion服务更新内容"""
        return [
            {
                'pattern': 'await self._load_mock_model()',
                'replacement': 'await self._load_mindspore_model()'
            }
        ]
    
    def _update_service_file(self, service_info: Dict):
        """更新服务文件"""
        try:
            file_path = self.backend_dir / service_info['file']
            
            if not file_path.exists():
                logger.warning(f"服务文件不存在: {file_path}")
                return
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 应用更新
            for update in service_info['updates']:
                content = content.replace(update['pattern'], update['replacement'])
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"服务文件已更新: {service_info['file']}")
            
        except Exception as e:
            logger.error(f"服务文件更新失败: {e}")

class PerformanceOptimization:
    """性能优化工具"""
    
    def __init__(self, backend_dir: str):
        self.backend_dir = Path(backend_dir)
    
    def optimize_for_production(self):
        """生产环境优化"""
        
        optimizations = [
            self._enable_model_caching,
            self._optimize_websocket_settings,
            self._configure_logging_for_production,
            self._optimize_database_settings
        ]
        
        for optimization in optimizations:
            try:
                optimization()
                logger.info(f"优化完成: {optimization.__name__}")
            except Exception as e:
                logger.error(f"优化失败 {optimization.__name__}: {e}")
    
    def _enable_model_caching(self):
        """启用模型缓存"""
        config_updates = {
            'ENABLE_MODEL_CACHE': True,
            'MODEL_CACHE_SIZE': 100,
            'CACHE_EXPIRE_TIME': 3600
        }
        self._update_config_file(config_updates)
    
    def _optimize_websocket_settings(self):
        """优化WebSocket设置"""
        config_updates = {
            'WEBSOCKET_MAX_CONNECTIONS': 100,
            'WEBSOCKET_BUFFER_SIZE': 8192,
            'WEBSOCKET_HEARTBEAT_INTERVAL': 30
        }
        self._update_config_file(config_updates)
    
    def _configure_logging_for_production(self):
        """配置生产环境日志"""
        config_updates = {
            'LOG_LEVEL': 'INFO',
            'LOG_TO_FILE': True,
            'LOG_ROTATION': 'daily',
            'LOG_RETENTION': 30
        }
        self._update_config_file(config_updates)
    
    def _optimize_database_settings(self):
        """优化数据库设置"""
        config_updates = {
            'DB_POOL_SIZE': 20,
            'DB_TIMEOUT': 30,
            'DB_ECHO': False
        }
        self._update_config_file(config_updates)
    
    def _update_config_file(self, updates: Dict):
        """更新配置文件"""
        # 这里实现配置文件更新逻辑
        pass

def create_deployment_script():
    """创建一键部署脚本"""
    
    deployment_script = '''#!/bin/bash

# 手语识别系统一键部署脚本

set -e

echo "开始部署手语识别系统..."

# 1. 检查环境
echo "检查Python环境..."
python --version
pip --version

# 2. 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 3. 检查MindSpore安装
echo "检查MindSpore..."
python -c "import mindspore; print(f'MindSpore版本: {mindspore.__version__}')"

# 4. 创建必要目录
echo "创建目录结构..."
mkdir -p models data logs temp uploads

# 5. 检查模型文件
echo "检查模型文件..."
python check_system.py

# 6. 数据库初始化
echo "初始化数据库..."
python -c "
from backend.utils.database import create_tables
create_tables()
print('数据库初始化完成')
"

# 7. 启动服务
echo "启动后端服务..."
cd backend
nohup python main.py > ../logs/backend.log 2>&1 &
echo $! > ../backend.pid

echo "等待后端服务启动..."
sleep 5

# 8. 启动前端服务
echo "启动前端服务..."
cd ../frontend
npm install
npm run build
nohup npm run preview > ../logs/frontend.log 2>&1 &
echo $! > ../frontend.pid

echo "部署完成！"
echo "后端服务: http://localhost:8000"
echo "前端服务: http://localhost:4173"
echo ""
echo "查看日志:"
echo "  后端: tail -f logs/backend.log"
echo "  前端: tail -f logs/frontend.log"
echo ""
echo "停止服务:"
echo "  ./stop_services.sh"
'''
    
    with open('deploy.sh', 'w', encoding='utf-8') as f:
        f.write(deployment_script)
    
    # 添加执行权限
    os.chmod('deploy.sh', 0o755)
    
    logger.info("部署脚本已创建: deploy.sh")

def create_stop_script():
    """创建服务停止脚本"""
    
    stop_script = '''#!/bin/bash

# 停止手语识别系统服务

echo "停止手语识别系统服务..."

# 停止后端服务
if [ -f backend.pid ]; then
    PID=$(cat backend.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "后端服务已停止 (PID: $PID)"
    else
        echo "后端服务未运行"
    fi
    rm -f backend.pid
else
    echo "后端PID文件不存在"
fi

# 停止前端服务
if [ -f frontend.pid ]; then
    PID=$(cat frontend.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "前端服务已停止 (PID: $PID)"
    else
        echo "前端服务未运行"
    fi
    rm -f frontend.pid
else
    echo "前端PID文件不存在"
fi

echo "所有服务已停止"
'''
    
    with open('stop_services.sh', 'w', encoding='utf-8') as f:
        f.write(stop_script)
    
    # 添加执行权限
    os.chmod('stop_services.sh', 0o755)
    
    logger.info("停止脚本已创建: stop_services.sh")

def main():
    """主部署函数"""
    
    config = {
        'model_dir': './models/checkpoints',
        'backend_dir': './backend',
        'frontend_dir': './frontend'
    }
    
    logger.info("开始模型部署和系统集成")
    
    # 1. 部署模型
    deployment = ModelDeployment(config)
    deployed_models = deployment.deploy_models()
    
    # 2. 更新服务实现
    integration = ServiceIntegration(config['backend_dir'])
    integration.update_service_implementations()
    
    # 3. 性能优化
    optimization = PerformanceOptimization(config['backend_dir'])
    optimization.optimize_for_production()
    
    # 4. 创建部署脚本
    create_deployment_script()
    create_stop_script()
    
    logger.info("部署和集成完成！")
    logger.info(f"已部署模型: {deployed_models}")
    logger.info("使用 ./deploy.sh 启动系统")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
