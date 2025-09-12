# GPU加速训练指南

本文档介绍如何在GPU服务器上进行TFNet模型的训练，相比CPU版本具有显著的性能提升。

## 📋 环境要求

### 硬件要求
- NVIDIA GPU（推荐RTX 3080/4080或更高）
- 至少8GB GPU显存
- 至少16GB系统内存
- 至少20GB可用磁盘空间

### 软件要求
- Linux操作系统
- NVIDIA驱动程序（版本 >= 460.x）
- CUDA 11.x 或 12.x
- conda环境管理器
- MindSpore GPU版本

## 🚀 快速开始

### 1. 激活环境
```bash
# 激活MindSpore GPU环境
conda activate mindspore-gpu
```

### 2. 检查GPU设置
```bash
# 进入训练目录
cd /root/shengteng/training

# 运行GPU系统测试
python test_gpu_setup.py
```

### 3. 快速启动训练
```bash
# 方法1: 使用Python快速启动脚本（推荐）
python quick_start_gpu.py

# 方法2: 使用Shell脚本
./start_training_gpu.sh

# 方法3: 直接运行GPU训练脚本
python train_tfnet_gpu.py --config configs/gpu_config.json
```

## 📁 GPU版本文件结构

```
training/
├── configs/
│   └── gpu_config.json          # GPU优化配置文件
├── train_tfnet_gpu.py           # GPU优化训练脚本
├── start_training_gpu.py        # GPU训练启动器（Python）
├── start_training_gpu.sh        # GPU训练启动器（Shell）
├── quick_start_gpu.py           # 快速启动脚本
├── test_gpu_setup.py           # GPU系统测试脚本
├── checkpoints_gpu/            # GPU训练检查点目录
├── logs_gpu/                   # GPU训练日志目录
└── output_gpu/                 # GPU训练输出目录
```

## ⚙️ GPU优化配置

GPU版本相比CPU版本的主要优化：

### 1. 硬件配置
- **设备目标**: GPU而非CPU
- **设备ID**: 指定GPU设备（默认为0）
- **批处理大小**: 增加到8（CPU版本为2）
- **工作进程**: 增加到4（CPU版本为1）

### 2. MindSpore优化
- **图模式**: 启用GRAPH_MODE获得更好性能
- **内存复用**: 启用内存优化
- **图内核优化**: 启用graph kernel加速
- **自动混合精度**: 可选启用（需要硬件支持）

### 3. 数据加载优化
- **预取大小**: 设置为2缓冲更多数据
- **最大行大小**: 优化内存使用
- **数据沉淀模式**: 启用以提高GPU利用率

## 📊 性能对比

| 配置项 | CPU版本 | GPU版本 | 提升倍数 |
|--------|---------|---------|----------|
| 批处理大小 | 2 | 8 | 4x |
| 工作进程数 | 1 | 4 | 4x |
| 预计训练速度 | 基准 | 10-20x | 10-20x |
| 内存使用 | 系统内存 | GPU显存 | - |

## 🔧 故障排除

### 1. 环境问题
```bash
# 检查conda环境
echo $CONDA_DEFAULT_ENV

# 检查MindSpore安装
python -c "import mindspore; print(mindspore.__version__)"

# 检查GPU可用性
nvidia-smi
```

### 2. 常见错误

#### GPU不可用
```
Error: GPU not available
```
**解决方案:**
1. 确认NVIDIA驱动已安装
2. 确认CUDA已安装
3. 确认MindSpore GPU版本已安装
4. 运行 `nvidia-smi` 检查GPU状态

#### 内存不足
```
Error: Out of GPU memory
```
**解决方案:**
1. 减少批处理大小（在配置文件中）
2. 减少模型大小（hidden_size参数）
3. 关闭其他GPU应用程序

#### 环境未激活
```
Warning: Not in mindspore-gpu environment
```
**解决方案:**
```bash
conda activate mindspore-gpu
```

### 3. 调试模式

如果遇到问题，可以启用调试模式：

1. 编辑 `configs/gpu_config.json`
2. 设置以下配置：
```json
{
    "gpu_optimization": {
        "enable_graph_mode": false,
        "enable_profiling": true,
        "enable_dump": true
    }
}
```

## 📈 性能调优

### 1. 批处理大小优化
根据GPU显存调整批处理大小：
- 8GB显存: batch_size = 4-6
- 12GB显存: batch_size = 6-10
- 16GB+显存: batch_size = 8-16

### 2. 工作进程数优化
根据CPU核心数调整：
- 4核CPU: num_workers = 2-4
- 8核CPU: num_workers = 4-6
- 16核+CPU: num_workers = 6-8

### 3. 学习率调整
GPU训练时由于批处理大小增大，可能需要调整学习率：
- 批处理大小增加4倍 → 学习率增加2倍
- 使用warmup策略避免训练不稳定

## 📝 监控训练

### 1. 实时监控
```bash
# 监控GPU使用率
watch -n 1 nvidia-smi

# 监控训练日志
tail -f training/logs_gpu/gpu_training_*.log
```

### 2. 检查训练进度
训练过程中会在以下位置保存文件：
- **检查点**: `training/checkpoints_gpu/`
- **日志文件**: `training/logs_gpu/`
- **最佳模型**: `training/checkpoints_gpu/best_model.ckpt`

## 🎯 预期结果

使用GPU训练，您应该看到：
1. **显著的速度提升**: 每个epoch时间减少10-20倍
2. **更大的批处理**: 能够处理更大的批处理大小
3. **更稳定的训练**: GPU内存管理更加稳定
4. **更好的模型性能**: 更大批处理可能带来更好的训练效果

## 💡 提示

1. **首次运行**: 建议先运行 `test_gpu_setup.py` 确保环境正确
2. **数据集位置**: 确保数据集在正确的路径 `data/CE-CSL/`
3. **磁盘空间**: 确保有足够空间存储检查点和日志
4. **备份重要数据**: 训练前备份重要的模型和数据

## 📞 支持

如果遇到问题：
1. 首先运行 `test_gpu_setup.py` 诊断
2. 检查 `training/logs_gpu/` 中的详细日志
3. 确认所有环境要求已满足
4. 尝试使用较小的配置（减少batch_size等）

祝您训练顺利！🚀
