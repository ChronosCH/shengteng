# Linux训练启动脚本使用指南

本目录包含两个Linux系统下的训练启动脚本：

## 脚本说明

### 1. start_training.sh - 完整版启动脚本

功能丰富的训练启动脚本，包含完整的环境检查和错误处理。

#### 特性
- 完整的环境检查（conda环境、项目结构、数据集、依赖包）
- 彩色输出和详细的错误信息
- 支持多种启动模式（训练、评估、检查）
- 自动日志记录
- 灵活的参数配置
- 错误恢复建议

#### 使用方法

```bash
# 给脚本添加执行权限
chmod +x training/start_training.sh

# 基本使用
./training/start_training.sh train                    # 开始训练
./training/start_training.sh eval                     # 运行评估
./training/start_training.sh check                    # 仅检查环境

# 高级使用
./training/start_training.sh train --config custom.json  # 使用自定义配置
./training/start_training.sh train --resume checkpoint.ckpt  # 从检查点恢复
./training/start_training.sh eval --model best_model.ckpt    # 指定模型评估
./training/start_training.sh train --skip-checks      # 跳过环境检查

# 查看帮助
./training/start_training.sh --help
```

### 2. quick_start.sh - 快速启动脚本

简化版脚本，适合经验丰富的用户快速启动训练。

#### 特性
- 最小化检查，快速启动
- 简洁的输出
- 支持基本操作

#### 使用方法

```bash
# 给脚本添加执行权限
chmod +x training/quick_start.sh

# 使用方法
./training/quick_start.sh          # 默认开始训练
./training/quick_start.sh train    # 开始训练
./training/quick_start.sh eval     # 运行评估
./training/quick_start.sh test     # 运行基础测试
```

## 环境要求

### 1. 系统要求
- Linux操作系统
- Bash shell (version 4.0+)
- Conda或Miniconda

### 2. Python环境
```bash
# 创建conda环境
conda create -n shengteng python=3.8

# 激活环境
conda activate shengteng

# 安装依赖
pip install mindspore opencv-python numpy
```

### 3. 项目结构
确保在项目根目录运行脚本，项目结构应包含：
```
project_root/
├── training/
│   ├── train_tfnet.py
│   ├── evaluator.py
│   ├── config_manager.py
│   ├── configs/
│   │   └── tfnet_config.json
│   └── start_training.sh
├── data/
│   └── CE-CSL/
│       ├── video/
│       └── label/
└── ...
```

## 常见问题解决

### 1. 权限问题
```bash
chmod +x training/start_training.sh
chmod +x training/quick_start.sh
```

### 2. Conda环境问题
```bash
# 如果conda命令不存在
export PATH="/path/to/conda/bin:$PATH"

# 或者手动初始化conda
source /path/to/conda/etc/profile.d/conda.sh
```

### 3. 环境变量问题
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export PATH="/home/username/anaconda3/bin:$PATH"
source ~/.bashrc
```

### 4. 依赖包问题
```bash
# 重新安装依赖
conda activate shengteng
pip install --upgrade mindspore opencv-python numpy
```

## 日志文件

训练和评估的日志文件会自动保存到：
- `training/logs/training_YYYYMMDD_HHMMSS.log` - 训练日志
- `training/logs/evaluation_YYYYMMDD_HHMMSS.log` - 评估日志

## 脚本输出说明

脚本使用彩色输出来区分不同类型的信息：
- 🔵 蓝色 [INFO] - 一般信息
- 🟢 绿色 [SUCCESS] - 成功操作
- 🟡 黄色 [WARNING] - 警告信息
- 🔴 红色 [ERROR] - 错误信息

## 自定义配置

可以通过修改 `training/configs/tfnet_config.json` 来自定义训练参数，或者使用 `--config` 参数指定自定义配置文件。

## 技术支持

如果遇到问题，请：
1. 检查错误信息和建议
2. 确认环境配置正确
3. 查看日志文件获取详细信息
4. 使用 `--help` 参数查看使用说明
