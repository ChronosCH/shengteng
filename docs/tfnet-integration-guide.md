# TFNet手语识别集成文档

## 概述

本项目将PyTorch版本的TFNet（Temporal Super-Resolution Network）成功迁移到MindSpore框架，并针对华为昇腾AI处理器进行了优化。该实现支持连续手语识别(CSLR)，使用CE-CSL数据集进行训练。

## 主要特性

### 🚀 性能优化
- **华为昇腾AI处理器支持**: 针对昇腾910/310系列处理器优化
- **MindSpore框架**: 从PyTorch完全迁移到MindSpore
- **分布式训练**: 支持多卡训练
- **混合精度训练**: 支持AMP加速训练

### 🧠 模型架构
- **TFNet**: 时序超分辨率网络，支持多尺度时序建模
- **MSTNet**: 多尺度时序网络变体
- **VAC**: 视觉注意力和上下文建模
- **Transformer编码器**: 支持相对位置编码的多头注意力

### 📊 数据处理
- **CE-CSL数据集支持**: 完整的数据预处理管道
- **视频帧提取**: 自动将视频转换为帧序列
- **数据增强**: 随机裁剪、翻转、时序重新缩放
- **词汇表构建**: 自动构建和管理手语词汇表

## 项目结构

```
training/
├── tfnet_mindspore.py          # TFNet模型MindSpore实现
├── cecsl_data_processor.py     # CE-CSL数据处理模块
├── train_tfnet_cecsl.py        # 训练主程序
├── tfnet_decoder.py           # CTC解码和评估模块
├── start_tfnet_training.sh    # Linux启动脚本
├── start_tfnet_training.bat   # Windows启动脚本
└── configs/
    └── tfnet_cecsl_config.json # 训练配置文件

backend/services/
└── diffusion_slp_service.py   # 集成手语识别和生成服务
```

## 快速开始

### 1. 环境准备

#### 基础环境
```bash
# Python 3.8+
python --version

# 安装MindSpore (昇腾版本)
pip install mindspore
# 或GPU版本: pip install mindspore-gpu
# 或CPU版本: pip install mindspore-cpu
```

#### 昇腾环境（可选）
```bash
# 检查昇腾设备
npu-smi info

# 设置环境变量
export DEVICE_TARGET=Ascend
export DEVICE_ID=0
```

#### 依赖包
```bash
pip install opencv-python imageio tqdm numpy
```

### 2. 数据准备

#### 下载CE-CSL数据集
1. 下载CE-CSL数据集（需要申请访问权限）
2. 将数据集放置在以下结构：

```
data/CE-CSL/
├── video/               # 原始视频文件
│   ├── train/
│   ├── dev/
│   └── test/
├── train.corpus.csv     # 训练标签
├── dev.corpus.csv       # 验证标签
└── test.corpus.csv      # 测试标签
```

#### 预处理视频数据
```python
from training.cecsl_data_processor import CECSLVideoProcessor

processor = CECSLVideoProcessor()
processor.batch_process_dataset(
    data_path="./data/CE-CSL/video",
    save_path="./data/CE-CSL/processed",
    max_frames=300
)
```

### 3. 训练模型

#### 使用脚本启动训练

**Linux/MacOS:**
```bash
cd training
chmod +x start_tfnet_training.sh
./start_tfnet_training.sh
```

**Windows:**
```cmd
cd training
start_tfnet_training.bat
```

#### 手动启动训练
```bash
cd training
python train_tfnet_cecsl.py --config configs/tfnet_cecsl_config.json --mode train
```

### 4. 测试模型
```bash
python train_tfnet_cecsl.py --config configs/tfnet_cecsl_config.json --mode test
```

## 配置说明

### 模型配置
```json
{
  "model_config": {
    "d_model": 512,              // 模型隐藏维度
    "n_heads": 8,                // 注意力头数
    "n_layers": 2,               // Transformer层数
    "module_choice": "TFNet",    // 模型类型: TFNet/MSTNet/VAC
    "hidden_size": 512,          // 隐藏层大小
    "blank_id": 0,               // CTC空白标记ID
    "dataset_name": "CE-CSL"     // 数据集名称
  }
}
```

### 训练配置
```json
{
  "training_config": {
    "batch_size": 4,             // 批大小
    "learning_rate": 0.0001,     // 学习率
    "epochs": 55,                // 训练轮数
    "warmup_steps": 1000,        // 预热步数
    "gradient_clip": 1.0,        // 梯度裁剪
    "kd_weight": 25.0           // 知识蒸馏权重
  }
}
```

### 硬件配置
```json
{
  "hardware_config": {
    "device_target": "Ascend",   // 设备类型: Ascend/GPU/CPU
    "device_id": 0,              // 设备ID
    "distributed": false,        // 是否分布式训练
    "amp_level": "O1"            // 混合精度级别
  }
}
```

## 性能优化

### 昇腾AI处理器优化
- **图模式执行**: 使用MindSpore图模式获得最佳性能
- **算子融合**: 自动优化计算图
- **内存管理**: 智能内存分配和回收
- **数据流水线**: 高效的数据加载和预处理

### 训练优化技巧
1. **批大小调整**: 根据显存大小调整batch_size
2. **学习率调度**: 使用余弦退火和线性预热
3. **梯度累积**: 大批量训练的内存优化
4. **检查点保存**: 定期保存最佳模型

## API使用

### 手语识别服务
```python
from backend.services.diffusion_slp_service import diffusion_slp_service

# 初始化服务
await diffusion_slp_service.initialize()

# 手语识别
result = await diffusion_slp_service.recognize_sign_language(
    video_frames=video_frames,  # numpy array: (seq_len, H, W, C)
    frame_rate=25
)

print(f"识别结果: {result['recognized_sentence']}")
print(f"置信度: {result['confidence']}")
```

### 批量识别
```python
# 批量处理多个视频
results = await diffusion_slp_service.batch_recognize_sign_language(
    video_batch=[video1, video2, video3]
)

for i, result in enumerate(results):
    print(f"视频{i}: {result['recognized_sentence']}")
```

## 评估指标

### 支持的评估指标
- **WER (Word Error Rate)**: 词错误率
- **句子准确率**: 完全匹配的句子比例
- **BLEU分数**: 序列相似度评估
- **推理时间**: 平均推理延迟

### 评估示例
```python
from training.tfnet_decoder import TFNetEvaluator

evaluator = TFNetEvaluator("./backend/models/vocab.json")

# 评估预测结果
results = evaluator.evaluate_predictions(
    predictions=model_outputs,
    ground_truths=labels,
    input_lengths=seq_lengths,
    decode_method='beam_search',
    beam_size=10
)

print(f"WER: {results['wer']['wer']:.2f}%")
print(f"句子准确率: {results['wer']['sentence_accuracy']:.2f}%")
```

## 故障排除

### 常见问题

#### 1. MindSpore安装问题
```bash
# 卸载并重新安装
pip uninstall mindspore
pip install mindspore

# 检查安装
python -c "import mindspore; print(mindspore.__version__)"
```

#### 2. 昇腾设备不可用
```bash
# 检查驱动
npu-smi info

# 设置环境变量
export ASCEND_DEVICE_ID=0
export DEVICE_TARGET=Ascend
```

#### 3. 内存不足
- 减小batch_size
- 减少max_sequence_length
- 启用梯度检查点

#### 4. 数据加载慢
- 增加num_workers
- 使用SSD存储
- 预处理数据到内存

### 性能调优

#### 训练加速
1. **使用混合精度**: 设置amp_level="O1"
2. **增大批大小**: 在内存允许的情况下
3. **数据并行**: 多GPU/NPU训练
4. **模型并行**: 大模型分割

#### 推理优化
1. **模型量化**: 减少模型大小和推理时间
2. **批量推理**: 同时处理多个样本
3. **缓存机制**: 缓存常用结果
4. **流水线**: 异步处理

## 参考资料

### 原始论文
- **TFNet**: "Continuous Sign Language Recognition via Temporal Super-Resolution Network" (2022)
- **CE-CSL**: Chinese Sign Language Dataset

### 技术文档
- [MindSpore官方文档](https://www.mindspore.cn/docs/)
- [昇腾AI处理器文档](https://www.hiascend.com/document)
- [连续手语识别技术综述](https://arxiv.org/abs/2204.05405)

### 开源项目
- [原始TFNet PyTorch实现](https://github.com/example/tfnet)
- [MindSpore模型库](https://gitee.com/mindspore/models)

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置
```bash
git clone <your-repo>
cd shengteng
pip install -r requirements.txt
```

### 代码规范
- 遵循PEP 8编码规范
- 添加适当的注释和文档
- 编写单元测试

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 更新日志

### v1.0.0 (2024-08-12)
- ✅ 完成TFNet从PyTorch到MindSpore的迁移
- ✅ 支持华为昇腾AI处理器
- ✅ 集成CE-CSL数据集处理
- ✅ 实现完整的训练和推理流程
- ✅ 添加性能优化和监控功能
