# TFNet 快速启动指南

## 🚀 一键启动

### Windows 用户
```cmd
# 1. 激活环境并启动训练
training\start_training.bat train

# 2. 或者运行评估
training\start_training.bat eval

# 3. 或者检查环境
training\start_training.bat check
```

### Linux/Mac 用户
```bash
# 1. 激活环境
conda activate shengteng

# 2. 启动训练
python training/start_training.py train

# 3. 或者运行评估
python training/start_training.py eval

# 4. 或者检查环境
python training/start_training.py check
```

## 📋 前置条件检查清单

### ✅ 环境要求
- [ ] Python 3.7+ 已安装
- [ ] Conda 环境管理器已安装
- [ ] `shengteng` conda 环境已创建并激活
- [ ] MindSpore CPU版本已安装
- [ ] OpenCV-Python 已安装
- [ ] NumPy 已安装

### ✅ 数据要求
- [ ] CE-CSL 数据集已下载
- [ ] 训练数据位于 `data/CE-CSL/video/train/`
- [ ] 验证数据位于 `data/CE-CSL/video/dev/`
- [ ] 测试数据位于 `data/CE-CSL/video/test/`
- [ ] 标签文件位于 `data/CE-CSL/label/` 目录

### ✅ 文件结构
```
项目根目录/
├── data/
│   └── CE-CSL/
│       ├── video/
│       │   ├── train/
│       │   ├── dev/
│       │   └── test/
│       └── label/
│           ├── train.csv
│           ├── dev.csv
│           └── test.csv
└── training/
    ├── start_training.bat      # Windows启动脚本
    ├── start_training.py       # Python启动脚本
    ├── train_tfnet.py         # 主训练脚本
    ├── evaluator.py           # 评估脚本
    └── configs/
        └── tfnet_config.json  # 配置文件
```

## 🔧 环境安装

### 1. 创建Conda环境
```bash
conda create -n shengteng python=3.8
conda activate shengteng
```

### 2. 安装依赖包
```bash
# 安装MindSpore CPU版本
pip install mindspore

# 安装其他依赖
pip install opencv-python numpy
```

### 3. 验证安装
```bash
python -c "import mindspore; print('MindSpore version:', mindspore.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## 📊 训练配置

### 默认配置（适合CPU训练）
- **批处理大小**: 2
- **学习率**: 0.0001
- **训练轮数**: 55
- **隐藏层大小**: 1024
- **设备**: CPU

### 自定义配置
1. 复制配置文件：
   ```bash
   cp training/configs/tfnet_config.json training/configs/my_config.json
   ```

2. 修改参数（推荐CPU优化设置）：
   ```json
   {
     "training": {
       "batch_size": 1,        # 减少内存使用
       "learning_rate": 0.0001,
       "num_epochs": 30        # 减少训练时间
     },
     "model": {
       "hidden_size": 512      # 减少模型复杂度
     }
   }
   ```

3. 使用自定义配置：
   ```bash
   python training/train_tfnet.py --config training/configs/my_config.json
   ```

## 📈 训练监控

### 训练日志
- 位置：`training/logs/training_YYYYMMDD_HHMMSS.log`
- 包含：损失值、WER指标、训练进度

### 模型检查点
- 最佳模型：`training/checkpoints/best_tfnet_model.ckpt`
- 当前模型：`training/checkpoints/current_tfnet_model.ckpt`

### 输出文件
- 词汇表：`training/output/vocabulary.json`
- 评估结果：`training/output/evaluation_results_*.json`

## 🔍 故障排除

### 常见错误及解决方案

1. **环境激活失败**
   ```
   Error: Failed to activate conda environment 'shengteng'
   ```
   **解决方案**：
   ```bash
   conda env list  # 检查环境是否存在
   conda create -n shengteng python=3.8  # 如果不存在则创建
   ```

2. **数据路径错误**
   ```
   Error: Training data not found
   ```
   **解决方案**：
   - 检查数据集是否正确解压到 `data/CE-CSL/` 目录
   - 确认文件夹结构正确

3. **内存不足**
   ```
   RuntimeError: out of memory
   ```
   **解决方案**：
   - 减少 `batch_size` 到 1
   - 减少 `hidden_size` 到 512
   - 减少 `max_frames` 到 200

4. **MindSpore导入错误**
   ```
   ImportError: No module named 'mindspore'
   ```
   **解决方案**：
   ```bash
   pip install mindspore
   ```

## 📞 技术支持

### 检查系统状态
```bash
# 运行完整的环境检查
python training/start_training.py check

# 运行基本功能测试
python training/test_basic.py

# 运行简单测试
python training/simple_test.py
```

### 获取帮助
```bash
# 查看训练脚本帮助
python training/train_tfnet.py --help

# 查看评估脚本帮助
python training/evaluator.py --help

# 查看启动脚本帮助
python training/start_training.py --help
```

## 🎯 性能优化建议

### CPU优化
1. **减少批处理大小**：设置 `batch_size = 1`
2. **降低模型复杂度**：设置 `hidden_size = 512`
3. **减少工作进程**：设置 `num_workers = 1`
4. **启用早停**：设置 `early_stopping_patience = 5`

### 内存优化
1. **限制视频长度**：设置 `max_frames = 200`
2. **减小图像尺寸**：设置 `crop_size = 224`
3. **梯度裁剪**：设置 `gradient_clip_norm = 1.0`

## 📝 使用示例

### 完整训练流程
```bash
# 1. 检查环境
python training/start_training.py check

# 2. 开始训练
python training/start_training.py train

# 3. 评估模型
python training/start_training.py eval
```

### 断点续训
```bash
python training/train_tfnet.py --resume training/checkpoints/current_tfnet_model.ckpt
```

### 自定义评估
```bash
python training/evaluator.py --model training/checkpoints/best_tfnet_model.ckpt
```

---

🎉 **恭喜！您已经完成了TFNet训练系统的设置。开始您的手语识别之旅吧！**
