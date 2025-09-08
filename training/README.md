# TFNet 连续手语识别训练系统

基于MindSpore框架的TFNet模型实现，专为CPU环境优化，支持CE-CSL数据集的连续手语识别训练。

## 目录结构

```
training/
├── README.md                 # 本文档
├── start_training.py         # Python启动脚本
├── start_training.bat        # Windows批处理启动脚本
├── train_tfnet.py           # 主训练脚本
├── evaluator.py             # 模型评估脚本
├── config_manager.py        # 配置管理器
├── tfnet_model.py           # TFNet模型定义
├── modules.py               # 基础模块
├── data_processor.py        # 数据处理模块
├── decoder.py               # CTC解码器
├── configs/                 # 配置文件目录
│   └── tfnet_config.json   # TFNet配置文件
├── checkpoints/             # 模型检查点目录
├── logs/                    # 训练日志目录
└── output/                  # 输出文件目录
```

## 环境要求

### 系统要求
- Windows 10/11 或 Linux
- Python 3.7+
- Conda 环境管理器

### 依赖包
- MindSpore (CPU版本)
- OpenCV-Python
- NumPy
- 其他标准库

### 数据要求
- CE-CSL数据集，包含：
  - `data/CE-CSL/video/train/` - 训练视频
  - `data/CE-CSL/video/dev/` - 验证视频
  - `data/CE-CSL/video/test/` - 测试视频
  - `data/CE-CSL/label/train.csv` - 训练标签
  - `data/CE-CSL/label/dev.csv` - 验证标签
  - `data/CE-CSL/label/test.csv` - 测试标签

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate shengteng

# 安装依赖（如果尚未安装）
pip install mindspore opencv-python numpy
```

### 2. 检查环境

```bash
# 使用Python脚本检查
python training/start_training.py check

# 或使用Windows批处理文件
training/start_training.bat check
```

### 3. 开始训练

```bash
# 使用Python脚本
python training/start_training.py train

# 或使用Windows批处理文件
training/start_training.bat train

# 或直接运行训练脚本
python training/train_tfnet.py --config training/configs/tfnet_config.json
```

### 4. 模型评估

```bash
# 使用Python脚本
python training/start_training.py eval

# 或使用Windows批处理文件
training/start_training.bat eval

# 或直接运行评估脚本
python training/evaluator.py --config training/configs/tfnet_config.json
```

## 配置说明

### 主要配置参数

配置文件位于 `training/configs/tfnet_config.json`，主要参数包括：

#### 数据集配置
```json
"dataset": {
    "name": "CE-CSL",
    "train_data_path": "data/CE-CSL/video/train",
    "valid_data_path": "data/CE-CSL/video/dev",
    "test_data_path": "data/CE-CSL/video/test",
    "train_label_path": "data/CE-CSL/label/train.csv",
    "valid_label_path": "data/CE-CSL/label/dev.csv",
    "test_label_path": "data/CE-CSL/label/test.csv",
    "crop_size": 224,
    "max_frames": 300
}
```

#### 模型配置
```json
"model": {
    "name": "TFNet",
    "hidden_size": 1024,
    "device_target": "CPU"
}
```

#### 训练配置
```json
"training": {
    "batch_size": 2,
    "learning_rate": 0.0001,
    "num_epochs": 55,
    "num_workers": 1,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "save_interval": 5,
    "eval_interval": 1,
    "early_stopping_patience": 10
}
```

### 自定义配置

1. 复制默认配置文件：
```bash
cp training/configs/tfnet_config.json training/configs/my_config.json
```

2. 修改配置参数

3. 使用自定义配置训练：
```bash
python training/train_tfnet.py --config training/configs/my_config.json
```

## 高级用法

### 断点续训

```bash
# 从最新检查点恢复训练
python training/train_tfnet.py --config training/configs/tfnet_config.json --resume training/checkpoints/current_tfnet_model.ckpt
```

### 指定模型评估

```bash
# 评估特定模型
python training/evaluator.py --config training/configs/tfnet_config.json --model training/checkpoints/best_tfnet_model.ckpt
```

### 批量处理

```bash
# 跳过环境检查（适用于自动化脚本）
python training/start_training.py train --skip-checks
```

## 输出文件

### 训练输出
- `training/checkpoints/best_tfnet_model.ckpt` - 最佳模型
- `training/checkpoints/current_tfnet_model.ckpt` - 当前模型
- `training/logs/training_YYYYMMDD_HHMMSS.log` - 训练日志
- `training/output/vocabulary.json` - 词汇表

### 评估输出
- `training/output/evaluation_results_YYYYMMDD_HHMMSS.json` - 评估结果

## 性能优化

### CPU优化建议
1. 调整批处理大小：根据内存情况调整 `batch_size`
2. 减少工作进程：设置 `num_workers` 为 1
3. 降低模型复杂度：减少 `hidden_size`
4. 早停机制：使用 `early_stopping_patience` 避免过拟合

### 内存优化
1. 限制最大帧数：调整 `max_frames`
2. 减小图像尺寸：调整 `crop_size`
3. 梯度累积：增加 `gradient_clip_norm`

## 故障排除

### 常见问题

1. **环境激活失败**
   ```
   Error: Failed to activate conda environment 'shengteng'
   ```
   解决方案：确保conda环境存在，运行 `conda env list` 检查

2. **数据路径错误**
   ```
   Error: Training data not found
   ```
   解决方案：检查数据集路径，确保CE-CSL数据集正确解压

3. **内存不足**
   ```
   RuntimeError: out of memory
   ```
   解决方案：减少batch_size或hidden_size

4. **MindSpore导入错误**
   ```
   ImportError: No module named 'mindspore'
   ```
   解决方案：安装MindSpore CPU版本

### 调试模式

启用详细日志：
```json
"logging": {
    "level": "DEBUG",
    "save_logs": true,
    "print_interval": 1
}
```

## 技术支持

如遇到问题，请检查：
1. 环境配置是否正确
2. 数据集是否完整
3. 配置文件是否有效
4. 日志文件中的错误信息

## 更新日志

- v1.0.0: 初始版本，支持TFNet模型训练和评估
- 基于MindSpore框架，优化CPU执行
- 支持CE-CSL数据集
- 包含完整的训练和评估流程
