#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CE-CSL手语识别服务启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入FastAPI应用
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动增强版CE-CSL手语识别服务...")
    print(f"📁 项目路径: {project_root}")
    print(f"🔧 Python路径: {sys.path[:3]}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # 禁用自动重载避免路径问题
        access_log=True
    )
