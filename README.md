# 🤖 SignAvatar Web - 手语识别与虚拟人播报系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.7+-orange.svg)](https://mindspore.cn)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于深度学习的实时手语识别与虚拟人播报系统，集成了多种先进技术，为听障人士提供全面的无障碍通信解决方案。

## ✨ 主要特性

### 🔍 核心功能
- **实时手语识别**: 基于CSLR（连续手语识别）技术，支持实时视频流手语识别
- **扩散模型手语生成**: 使用Diffusion SLP技术生成自然流畅的手语动作序列
- **虚拟人播报**: 3D虚拟人实时播报手语内容，支持多种情感和语速
- **多模态传感器融合**: 集成EMG、IMU和视觉传感器，提高识别精度

### 🛡️ 隐私保护
- **差分隐私**: 保护用户数据隐私的同时保持模型性能
- **数据匿名化**: 智能图像/视频匿名化处理
- **联邦学习**: 分布式训练，数据不出本地

### ♿ 无障碍支持
- **触觉反馈**: 支持触觉设备和盲文显示器
- **语义反馈**: 智能语义触觉反馈系统
- **可访问性优化**: 完整的无障碍界面设计

### 🚀 技术亮点
- **高性能**: 基于MindSpore深度学习框架
- **实时处理**: WebSocket实时通信，低延迟响应
- **可扩展**: 微服务架构，支持水平扩展
- **监控完备**: 集成Prometheus+Grafana监控体系

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   摄像头    │ │  虚拟人显示  │ │  控制面板    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket
┌─────────────────────────────────────────────────────────────┐
│                      后端API服务                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  手语识别    │ │  手语生成    │ │  用户管理    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  隐私保护    │ │  触觉反馈    │ │  联邦学习    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      基础设施                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   数据库     │ │    缓存      │ │    监控      │           │
│  │  (SQLite)   │ │  (Redis)    │ │(Prometheus) │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 📋 系统要求

- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.11 或更高版本
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间
- **GPU**: 可选，推荐用于模型推理加速

### 🔧 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/your-username/signavatar-web.git
cd signavatar-web
```

#### 2. 环境准备
```bash
# 安装Python依赖
pip install -r requirements.txt

# 初始化项目环境
python start.py init
```

#### 3. 配置环境
```bash
# 复制并编辑配置文件
cp .env.example .env
# 编辑 .env 文件，设置必要的配置参数
```

#### 4. 启动服务

**开发环境**:
```bash
# 启动开发服务器
python start.py start --reload

# 或者使用部署脚本
./deploy.sh -e development deploy
```

**生产环境**:
```bash
# 使用Docker Compose启动
docker-compose up -d

# 或者使用部署脚本
./deploy.sh -e production deploy
```

### 🌐 访问应用

- **主应用**: http://localhost:3000
- **API文档**: http://localhost:8000/api/docs
- **监控面板**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📖 使用指南

### 🎯 基本使用

1. **手语识别**
   - 打开摄像头权限
   - 在镜头前进行手语动作
   - 系统实时识别并显示文本结果

2. **手语生成**
   - 在文本框输入要转换的文字
   - 选择情感和语速
   - 点击生成，观看虚拟人手语播报

3. **设置个性化**
   - 登录账户管理个人偏好
   - 调整识别敏感度
   - 配置无障碍选项

### 🔌 API使用

**手语识别API**:
```python
import asyncio
import websockets
import json

async def sign_recognition():
    uri = "ws://localhost:8000/ws/sign-recognition"
    async with websockets.connect(uri) as websocket:
        # 发送关键点数据
        data = {
            "type": "landmarks",
            "payload": {
                "landmarks": [[0.1, 0.2, 0.3], ...],
                "timestamp": 1234567890.0,
                "frame_id": 1
            }
        }
        await websocket.send(json.dumps(data))
        
        # 接收识别结果
        result = await websocket.recv()
        print(json.loads(result))

asyncio.run(sign_recognition())
```

**手语生成API**:
```python
import requests

response = requests.post("http://localhost:8000/api/diffusion/generate", 
    json={
        "text": "你好，很高兴见到你",
        "emotion": "happy",
        "speed": "normal"
    }
)
result = response.json()
print(result["data"]["keypoints"])
```

## ⚙️ 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DEBUG` | 调试模式 | `true` |
| `SECRET_KEY` | JWT密钥 | `your-secret-key` |
| `DATABASE_URL` | 数据库地址 | `sqlite:///./data/signavatar.db` |
| `REDIS_HOST` | Redis主机 | `localhost` |
| `CSLR_MODEL_PATH` | CSLR模型路径 | `models/cslr_model.mindir` |
| `DIFFUSION_MODEL_PATH` | Diffusion模型路径 | `models/diffusion_slp.mindir` |

### 模型文件

系统需要以下预训练模型文件（请联系开发团队获取）：

- `models/cslr_model.mindir` - CSLR识别模型
- `models/diffusion_slp.mindir` - Diffusion生成模型  
- `models/text_encoder.mindir` - 文本编码器
- `models/federated_slr.mindir` - 联邦学习模型
- `models/vocab.json` - 词汇表文件

## 🧪 开发指南

### 项目结构
```
signavatar-web/
├── backend/                 # 后端服务
│   ├── api/                # API路由
│   ├── services/           # 业务服务
│   ├── utils/              # 工具模块
│   └── main.py            # 主应用
├── frontend/               # 前端应用
│   ├── src/               # 源代码
│   └── public/            # 静态资源
├── models/                # AI模型文件
├── docs/                  # 文档
├── tests/                 # 测试文件
├── docker-compose.yml     # Docker配置
├── requirements.txt       # Python依赖
└── README.md             # 项目说明
```

### 运行测试
```bash
# 运行基本测试
python test_system.py

# 运行完整测试套件
python start.py test

# 运行特定类型测试
python start.py test --test-type unit
```

### 代码贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📊 监控与运维

### 系统监控

- **性能指标**: CPU、内存、网络使用情况
- **应用指标**: 请求量、响应时间、错误率
- **业务指标**: 识别准确率、用户活跃度

### 日志管理

```bash
# 查看实时日志
./deploy.sh logs

# 查看特定服务日志
docker-compose logs -f backend

# 查看系统状态
./deploy.sh status
```

### 备份恢复

```bash
# 备份数据
./deploy.sh backup

# 恢复数据（手动操作）
cp backups/20240806_120000/signavatar.db data/
```

## 🤝 技术支持

### 常见问题

**Q: 模型加载失败怎么办？**
A: 请确认模型文件路径正确，且文件完整。检查 `models/` 目录下是否有所需的 `.mindir` 文件。

**Q: WebSocket连接失败？**
A: 检查防火墙设置，确保8000端口可访问。在开发环境中，请确认后端服务正在运行。

**Q: 识别准确率较低？**
A: 可以尝试调整光照条件，确保手部清晰可见，或者在设置中调整识别敏感度。

### 联系我们

- **项目主页**: https://github.com/your-username/signavatar-web
- **问题反馈**: https://github.com/your-username/signavatar-web/issues
- **邮箱支持**: support@signavatar.com
- **技术文档**: https://docs.signavatar.com

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [MindSpore](https://mindspore.cn) 提供深度学习框架支持
- 感谢 [MediaPipe](https://mediapipe.dev) 提供手部关键点检测技术
- 感谢所有为无障碍技术发展做出贡献的开发者和研究者

---

**让技术连接每一个人，让沟通无障碍** 💙