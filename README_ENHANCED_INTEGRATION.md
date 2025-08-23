# 增强版CE-CSL手语识别模型Web集成

## 概述

本项目成功将训练好的增强版CE-CSL手语识别模型集成到Web应用中，提供实时手语识别服务。

## 训练模型信息

- **模型文件**: `training/output/enhanced_cecsl_final_model.ckpt`
- **词汇表**: `training/output/enhanced_vocab.json`
- **训练历史**: `training/output/enhanced_training_history.json`
- **训练脚本**: `training/enhanced_cecsl_trainer.py`

## 模型架构

- **特征提取**: 改进的密集网络层，支持图像特征提取
- **时序建模**: 双向LSTM，层数可配置
- **注意力机制**: 自注意力机制用于关键帧聚合
- **分类器**: 多层全连接网络，支持dropout正则化
- **优化**: 学习率调度、早停、数据增强等训练策略

## 系统要求

### 基础环境
- Python 3.8+
- MindSpore 2.0+ (可选，未安装时使用模拟模式)
- FastAPI + Uvicorn
- Node.js 16+ (用于前端开发服务器)

### Python依赖
```bash
pip install fastapi uvicorn mindspore numpy
```

## 快速启动

### 1. 激活环境
```bash
conda activate shengteng
```

### 2. 一键启动（推荐）
```bash
start_enhanced_app.bat
```

这会自动：
- 检查环境和依赖
- 启动后端服务 (端口8000)
- 启动前端服务 (端口5173)
- 运行集成测试
- 打开测试页面

### 3. 手动启动

#### 启动后端服务
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 启动前端（可选）
```bash
cd frontend
npm install
npm run dev
```

## API接口

### 1. 模型预测
```http
POST /api/enhanced-cecsl/test
Content-Type: application/json

{
    "landmarks": [[...], [...], ...],  // 关键点数据
    "description": "测试描述"
}
```

### 2. 服务统计
```http
GET /api/enhanced-cecsl/stats
```

### 3. 健康检查
```http
GET /api/health
```

## 测试页面

访问 `http://localhost:5173/enhanced-cecsl-test.html` 或直接打开 `frontend/enhanced-cecsl-test.html`

功能包括：
- 服务状态监控
- 单次/批量预测测试
- 测试历史记录
- 实时统计信息

## 项目结构

```
├── backend/
│   ├── main.py                     # 主应用入口
│   ├── services/
│   │   ├── enhanced_cecsl_service.py  # 增强版CE-CSL服务
│   │   └── ...
│   └── utils/
│       ├── config.py               # 配置管理
│       └── ...
├── training/
│   ├── enhanced_cecsl_trainer.py   # 训练脚本
│   └── output/
│       ├── enhanced_cecsl_final_model.ckpt  # 训练好的模型
│       ├── enhanced_vocab.json              # 词汇表
│       └── enhanced_training_history.json   # 训练历史
├── frontend/
│   ├── enhanced-cecsl-test.html    # 测试页面
│   └── ...
├── test_enhanced_integration.py    # 集成测试脚本
├── start_enhanced_app.bat         # 一键启动脚本
└── README_ENHANCED_INTEGRATION.md # 本文档
```

## 配置说明

模型路径配置在 `backend/utils/config.py`:

```python
# CSLR模型设置 - 更新为新训练的模型
CSLR_MODEL_PATH: str = "training/output/enhanced_cecsl_final_model.ckpt"
CSLR_VOCAB_PATH: str = "training/output/enhanced_vocab.json"
```

## 集成测试

运行完整的集成测试：

```bash
python test_enhanced_integration.py
```

测试内容：
- ✅ 模型文件检查
- ✅ 后端健康检查  
- ✅ 服务统计信息
- ✅ 模型预测测试

## 功能特性

### 后端服务
- **模型加载**: 自动加载训练好的checkpoint模型
- **预处理**: 关键点数据预处理和特征提取
- **推理优化**: 支持异步推理和缓存机制
- **错误处理**: 完善的错误处理和降级机制
- **监控**: 实时统计和性能监控

### 前端界面
- **实时测试**: 支持单次和批量预测测试
- **可视化**: 预测结果可视化展示
- **历史记录**: 测试历史记录和统计
- **响应式**: 支持移动端访问

### 数据流程
1. **输入**: MediaPipe关键点数据 (543个点 × 3坐标)
2. **预处理**: 数据归一化和特征映射
3. **模型推理**: 增强版CE-CSL模型预测
4. **后处理**: CTC解码和置信度计算
5. **输出**: 预测文本、置信度、手势序列

## 性能指标

- **推理速度**: ~20-50ms (CPU模式)
- **内存占用**: ~200-500MB
- **并发支持**: 异步处理，支持多用户
- **准确率**: 根据训练数据集表现

## 故障排除

### 1. 模型文件不存在
```
错误: 模型文件不存在: training/output/enhanced_cecsl_final_model.ckpt
解决: 请先运行训练脚本生成模型
```

### 2. MindSpore未安装
```
警告: MindSpore未安装，将使用模拟推理
解决: pip install mindspore（可选）
```

### 3. 端口冲突
```
错误: [Errno 10048] Only one usage of each socket address
解决: 修改端口或关闭占用进程
```

### 4. 权限问题
```
错误: Permission denied
解决: 以管理员身份运行或检查文件权限
```

## 扩展开发

### 添加新模型
1. 在 `services/` 目录创建新服务文件
2. 实现 `predict_from_landmarks` 接口
3. 在 `main.py` 中注册服务
4. 添加相应的API端点

### 自定义预处理
1. 修改 `enhanced_cecsl_service.py` 中的 `_preprocess_landmarks` 方法
2. 根据具体需求调整数据格式转换

### 集成其他模型
1. 继承或参考 `EnhancedCECSLService` 类
2. 实现模型特定的加载和推理逻辑
3. 保持统一的接口规范

## 更新日志

### v1.0.0 (2025-08-22)
- ✨ 成功集成增强版CE-CSL模型
- ✨ 实现Web API接口
- ✨ 添加测试页面和集成测试
- ✨ 支持实时预测和批量测试
- ✨ 完善的错误处理和监控

## 联系信息

如有问题或建议，请查看：
- 项目文档: `docs/` 目录
- 训练报告: `training_improvement_report.md`
- 架构报告: `训练架构优化完成报告.md`

---

🎉 **恭喜！** 增强版CE-CSL手语识别模型已成功集成到Web应用中！
