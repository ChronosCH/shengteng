# 🤖 手语识别与学习系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.7+-orange.svg)](https://mindspore.cn)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于深度学习的实时手语识别与学习系统，集成了多种先进技术，为听障人士提供全面的无障碍通信解决方案。

## ✨ 主要特性

### 🔍 核心功能
- **实时手语识别**: 基于 Mind-VAC CSLR（连续手语识别）管线，支持 MindSpore 推理与实时视频流手语识别
- **扩散模型手语生成**: 使用Diffusion SLP技术生成自然流畅的手语动作序列
- **多模态传感器融合**: 集成EMG、IMU和视觉传感器，提高识别精度
- **智能学习训练**: 个性化学习路径，系统化手语学习

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

┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   摄像头    │ │  手语显示    │ │  控制面板    │           │
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
git clone https://github.com/ChronosCH/shengteng.git
cd shengteng
```

#### 2. 环境准备
```bash
# 激活 Mind-VAC 推理所需的 Conda 环境
conda activate mind

# 安装 Python 依赖
pip install -r requirements.txt
```

> 提示：依赖列表已包含 `requests`，用于 Mind-VAC 通义千问 API 客户端；如遇缺失，请重新执行上述安装命令。
>
> 若尚未创建 `mind` 环境，可参考以下示例：`conda create -n mind python=3.11 mindspore -c mindspore`，然后再执行上述步骤。

#### 3. 配置环境
```bash
# 复制并编辑配置文件
cp .env.example .env
# 编辑 .env 文件，设置必要的配置参数
```

#### 4. 启动服务

**开发环境**:
```bash
# 启动后端（FastAPI）
python backend/main.py

# 启动前端（Vite）
cd frontend
npm install
npm run dev
```

> 首次启动后端时，请关注日志：若看到 “Mind-VAC 模型与资源加载完成”，说明 CSLR 引擎已正常工作；若出现依赖缺失提示，可根据日志补齐可选组件。

**生产环境**:
```bash
# 使用 Uvicorn 启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2

# 使用 Docker Compose 一键部署
docker-compose up -d
```

### 🌐 访问应用

- **前端开发环境**: http://localhost:5173
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
| `CSLR_MODEL_PATH` | Mind-VAC CSLR 模型权重路径 | `mind_vac/slr_mindspore.ckpt` |
| `CSLR_VOCAB_PATH` | Mind-VAC 词典（支持 .json/.npy） | `mind_vac/gloss_dict.npy` |
| `MINDVAC_ENABLED` | 是否启用 Mind-VAC 管线 | `true` |
| `MINDVAC_DEVICE` | Mind-VAC 推理设备 | `CPU` |
| `DIFFUSION_MODEL_PATH` | Diffusion模型路径 | `models/diffusion_slp.mindir` |

### 模型文件

系统需要以下关键模型与资源文件（请联系开发团队获取）：

- `mind_vac/slr_mindspore.ckpt` - Mind-VAC CSLR MindSpore 权重
- `mind_vac/gloss_dict.npy` - Mind-VAC 词典（numpy/pickle 序列化，亦支持 JSON 版本）
- `mind_vac/output_dir/` - Mind-VAC 推理输出目录（运行时自动创建）
- `models/diffusion_slp.mindir` - Diffusion手语生成模型（可选）
- `models/text_encoder.mindir` - 文本编码器（可选）
- `models/federated_slr.mindir` - 联邦学习模型（可选）

#### Mind-VAC 资源准备

1. 确保上述 Mind-VAC 权重与词典文件位于 `mind_vac/` 目录。
2. 如果使用 `.npy` 词典，保持原始 numpy/pickle 编码即可，系统会自动转换；如需自定义，可提供 UTF-8 JSON 映射。
3. 若启用通义千问增强功能，请在运行前设置 `DASHSCOPE_API_KEY` 环境变量。
4. 如果未安装 MediaPipe 或 PyJWT，后端会自动降级并给出提示日志，无需手动禁用。

#### 模块依赖说明

- **MediaPipe（可选）**: 未安装时仍可使用 Mind-VAC CSLR 识别，日志会显示降级提示。
- **PyJWT/Passlib（可选）**: 认证服务默认关闭，可按需安装并启用。
- **requests**: Mind-VAC 通义千问客户端必需，已在 `requirements.txt` 中声明。

## 🏋️ 模型训练

### 训练环境准备

在开始模型训练之前，请确保满足以下要求：

#### 硬件要求
- **CPU**: Intel i5/AMD Ryzen 5 或更高
- **内存**: 至少 16GB RAM（推荐 32GB）
- **GPU**: 可选，推荐 NVIDIA GTX 1660 或更高（用于加速训练）
- **存储**: 至少 50GB 可用空间

#### 软件依赖
```bash
# 安装训练相关依赖
pip install -r requirements.txt

# 验证环境配置
cd training
python check_env.py
```

### 数据集准备

#### CE-CSL 数据集
本项目使用 CE-CSL（Chinese Sign Language）数据集进行训练：

```bash
# 数据集目录结构
data/
├── CE-CSL/                    # 主数据集
│   ├── videos/               # 原始视频文件
│   │   ├── 001_你好_001.mp4
│   │   ├── 002_谢谢_001.mp4
│   │   └── ...
│   ├── labels/               # 标签文件
│   │   ├── train_labels.csv
│   │   └── test_labels.csv
│   └── corpus.txt           # 词汇表
├── CS-CSL/                   # 扩展数据集（可选）
└── processed/               # 预处理后的数据
```

#### 数据预处理
```bash
cd training

# 1. 完整数据预处理
python complete_preprocessing.py

# 2. 数据集分析
python analyze_full_dataset.py

# 3. 数据验证
python inspect_cecsl_data.py
python validate_labels.py

# 4. 标签清理（如有需要）
python label_cleaner.py
```

### 训练流程

#### 快速开始训练
```bash
cd training

# 使用默认配置开始训练
python train.py --data_root ../data/CE-CSL --epochs 100

# 指定GPU训练
python train.py --data_root ../data/CE-CSL --epochs 100 --device gpu

# 自定义配置训练
python train.py --data_root ../data/CE-CSL \
                --epochs 200 \
                --batch_size 2 \
                --learning_rate 2e-4
```

#### 高级训练配置

创建自定义训练脚本：

```python
# custom_training.py
from training.enhanced_cecsl_trainer import EnhancedCECSLConfig, EnhancedCECSLTrainer

# 创建配置
config = EnhancedCECSLConfig()

# 模型配置
config.vocab_size = 1000
config.d_model = 256          # 增加模型维度
config.n_layers = 4           # 增加层数
config.dropout = 0.2

# 训练配置
config.batch_size = 4         # 根据显存调整
config.learning_rate = 1e-4
config.weight_decay = 1e-3
config.epochs = 300
config.warmup_epochs = 20

# 数据配置
config.data_root = "data/CE-CSL"
config.max_sequence_length = 128
config.image_size = (224, 224)

# 数据增强配置
config.enable_augmentation = True
config.augmentation_prob = 0.8

# 创建训练器
trainer = EnhancedCECSLTrainer(config)

# 加载数据和构建模型
trainer.load_data()
trainer.build_model()

# 开始训练
trainer.train()
```

#### 训练监控

训练过程中可以监控以下指标：

```bash
# 查看训练日志
tail -f training/logs/training.log

# 查看训练进度
python -c "
from training.enhanced_cecsl_trainer import EnhancedCECSLTrainer
trainer = EnhancedCECSLTrainer.load_from_checkpoint('training/checkpoints/latest.ckpt')
print(f'当前训练进度: {trainer.current_epoch}/{trainer.config.epochs}')
print(f'最佳准确率: {trainer.best_accuracy:.4f}')
"
```

### 训练优化策略

#### 内存优化
```python
# 小批次训练 + 梯度累积
config.batch_size = 1
config.gradient_accumulation_steps = 8

# 混合精度训练
config.use_mixed_precision = True

# 动态调整序列长度
config.dynamic_sequence_length = True
```

#### 性能优化
```python
# 数据预加载
config.num_workers = 4
config.prefetch_factor = 2

# 模型并行
config.model_parallel = True

# 知识蒸馏
config.use_knowledge_distillation = True
config.teacher_model_path = "models/teacher_model.ckpt"
```

### 训练结果评估

#### 模型评估
```bash
cd training

# 评估训练好的模型
python -c "
from enhanced_cecsl_trainer import EnhancedCECSLTrainer

# 加载最佳模型
trainer = EnhancedCECSLTrainer.load_checkpoint('checkpoints/best_model.ckpt')

# 在测试集上评估
results = trainer.evaluate()
print(f'测试准确率: {results[\"accuracy\"]:.4f}')
print(f'测试损失: {results[\"loss\"]:.4f}')
print(f'F1分数: {results[\"f1_score\"]:.4f}')
"
```

#### 性能基准
预期的训练性能指标：

| 数据集 | 训练时间 | 验证准确率 | 测试准确率 | 模型大小 |
|--------|----------|------------|------------|----------|
| CE-CSL | 4-6小时 | 85-90% | 82-87% | ~50MB |
| CS-CSL | 8-12小时 | 88-92% | 85-90% | ~50MB |

### 模型部署

#### 导出训练好的模型
```bash
cd training

# 导出为推理格式
python -c "
from enhanced_cecsl_trainer import EnhancedCECSLTrainer

trainer = EnhancedCECSLTrainer.load_checkpoint('checkpoints/best_model.ckpt')
trainer.export_model('exports/cslr_model.mindir')
print('模型导出完成')
"

# 复制到模型目录
cp exports/cslr_model.mindir ../models/
```

#### 模型版本管理
```bash
# 创建模型版本标签
cd training/checkpoints
mkdir -p versions/v1.0
cp best_model.ckpt versions/v1.0/
cp ../configs/training_config.json versions/v1.0/

# 记录版本信息
echo "训练时间: $(date)" > versions/v1.0/info.txt
echo "数据集: CE-CSL" >> versions/v1.0/info.txt
echo "准确率: 85.6%" >> versions/v1.0/info.txt
```

### 训练故障排除

#### 常见问题及解决方案

**Q: 训练过程中出现内存不足错误？**
```bash
# 解决方案：
1. 减小 batch_size
2. 启用梯度累积
3. 使用混合精度训练
4. 减少序列长度

# 示例配置
config.batch_size = 1
config.gradient_accumulation_steps = 4
config.use_mixed_precision = True
config.max_sequence_length = 64
```

**Q: 模型不收敛或准确率很低？**
```bash
# 解决方案：
1. 检查数据预处理
2. 调整学习率
3. 增加数据增强
4. 检查标签正确性

# 调试命令
python validate_labels.py
python analyze_full_dataset.py
```

**Q: 训练速度太慢？**
```bash
# 解决方案：
1. 启用GPU训练
2. 增加批次大小
3. 使用数据并行
4. 优化数据加载

# 性能优化
config.num_workers = 8
config.pin_memory = True
config.non_blocking = True
```

### TFNet 模型训练

本项目还集成了 TFNet 技术用于高精度手语识别：

```bash
# TFNet 训练
cd training

# 训练 TFNet 模型
python tfnet_mindspore.py --config configs/tfnet_config.yaml

# 使用 TFNet 解码器
python tfnet_decoder.py --model_path checkpoints/tfnet_model.ckpt
```

详细的训练流程和参数说明请参考 [`training/训练流程说明.md`](training/训练流程说明.md) 文档。

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
├── training/               # 模型训练
│   ├── 训练流程说明.md      # 训练文档
│   ├── train.py           # 训练入口
│   ├── enhanced_cecsl_trainer.py  # 增强训练器
│   ├── complete_preprocessing.py  # 数据预处理
│   ├── analyze_full_dataset.py    # 数据分析
│   ├── check_env.py       # 环境检查
│   ├── tfnet_mindspore.py # TFNet实现
│   ├── cache/             # 训练缓存
│   ├── checkpoints/       # 模型检查点
│   ├── configs/           # 训练配置
│   └── output/            # 训练输出
├── models/                # AI模型文件
├── data/                  # 训练数据集
│   ├── CE-CSL/           # CE-CSL数据集
│   └── CS-CSL/           # CS-CSL数据集
├── docs/                  # 文档
├── tests/                 # 测试文件
├── docker-compose.yml     # Docker配置
├── requirements.txt       # Python依赖
├── requirements-tfnet.txt # TFNet依赖
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

# 运行训练相关测试
cd training
python check_env.py                    # 检查训练环境
python analyze_full_dataset.py         # 分析数据集
python validate_labels.py              # 验证标签

# 运行集成测试
cd tests
python test_enhanced_integration.py    # 增强集成测试
python test_tfnet_integration.py       # TFNet集成测试
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

**Q: 训练过程中内存不足？**
A: 减小 batch_size，启用梯度累积，或使用混合精度训练。详见 [训练流程说明](training/训练流程说明.md)。

**Q: 如何开始模型训练？**
A: 首先运行 `python training/check_env.py` 检查环境，然后使用 `python training/train.py` 开始训练。

**Q: 训练数据集在哪里获取？**
A: CE-CSL数据集需要单独下载，请将数据放置在 `data/CE-CSL/` 目录下。详细的数据准备步骤请参考训练文档。

**Q: 如何监控训练进度？**
A: 可以查看 `training/logs/` 目录下的日志文件，或使用训练器提供的监控接口。

**Q: 训练完成后如何部署模型？**
A: 使用训练器的 `export_model()` 方法导出模型，然后复制到 `models/` 目录下替换现有模型。

### 联系我们

- **项目主页**: https://github.com/ChronosCH/shengteng
- **问题反馈**: https://github.com/ChronosCH/shengteng/issues
- **邮箱支持**: 请通过 Issues 联系维护者
- **技术文档**: 敬请期待

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [MindSpore](https://mindspore.cn) 提供深度学习框架支持
- 感谢 [MediaPipe](https://mediapipe.dev) 提供手部关键点检测技术
- 感谢所有为无障碍技术发展做出贡献的开发者和研究者

---

**让技术连接每一个人，让沟通无障碍** 💙