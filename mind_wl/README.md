# WLASL MindSpore版本 - CPU推理

本目录包含WLASL手语识别项目的MindSpore实现，支持CPU推理。

**⚠️ 重要提示：当前版本存在权重转换问题**

由于PyTorch和MindSpore在BatchNorm层的参数命名差异（`running_mean` vs `moving_mean`等），直接转换会导致228个参数无法正确加载，包括所有BatchNorm的统计量和可学习参数。这会导致推理结果完全不准确。

**已知问题：**
- ❌ BatchNorm参数名称不匹配
- ❌ 228/total参数未加载
- ❌ 推理结果不准确（所有视频预测为同一类别）

**解决方案：**
1. **推荐**：使用PyTorch版本进行推理
   - 位置：`/root/WLASL/simple_inference.py`
   - 文档：`/root/WLASL/CPU_README.md`
   - 状态：✅ 已验证，结果正确

2. **进阶**：手动修复参数映射（需要深入了解两个框架）
   - 需要创建完整的参数名称映射表
   - 需要处理不同的参数格式
   - 工作量较大，不推荐新手尝试

本MindSpore版本仅作为**框架转换示例**和**学习参考**，不建议用于实际推理任务。

## 📁 目录结构

```
mind_wl/
├── README.md                    # 本文件
├── inference_mindspore.py       # 推理脚本
├── convert_weights.py           # 权重转换工具
├── models/
│   └── i3d_mindspore.py        # I3D模型MindSpore实现
└── preprocess/
    └── (将自动链接到主项目的预处理文件)
```

## 🚀 快速开始

### 第一步：安装MindSpore

```bash
# 创建MindSpore环境
conda create -n mindspore_wlasl python=3.7 -y
conda activate mindspore_wlasl

# 安装MindSpore CPU版本
pip install mindspore==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用官方源
# pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/MindSpore/cpu/x86_64/mindspore-2.0.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install opencv-python numpy
```

### 第二步：验证安装

```bash
python -c "import mindspore; print('MindSpore版本:', mindspore.__version__)"
```

预期输出：`MindSpore版本: 2.0.0`

### 第三步：测试视频加载

不需要模型权重，先测试视频加载功能：

```bash
cd mind_wl
python inference_mindspore.py --test-only
```

这将：
- ✅ 测试视频加载
- ✅ 测试张量转换
- ✅ 测试简单模型推理
- ✅ 验证MindSpore环境

### 第四步：准备模型权重

#### 方案A：转换PyTorch权重（推荐）

如果你已有PyTorch的预训练权重：

```bash
# 1. 在同一环境安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 运行转换脚本
python convert_weights.py \
  ../code/I3D/archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt \
  weights/i3d_wlasl2000.ckpt
```

#### 方案B：直接下载MindSpore权重

如果有提供的MindSpore格式权重，直接放到 `weights/` 目录。

### 第五步：运行完整推理

```bash
python inference_mindspore.py
```

将对 `../test/` 目录下的所有视频进行推理。

## 📊 输出示例

### 测试模式输出

```
============================================================
MindSpore 视频加载测试
============================================================
使用设备: CPU
MindSpore版本: 2.0.0

找到 6 个视频文件:
  1. 1723591810067-CLEAR.mp4
  2. 2592423368318-BAT.mp4
  ...

============================================================
测试第一个视频的加载...
============================================================
正在加载视频: ../test/1723591810067-CLEAR.mp4
  - 总帧数: 67
  - FPS: 29.92
  - 成功采样 67 帧 (间隔: 1)

✓ 视频加载成功!
  - 张量形状: (1, 3, 67, 224, 224)
  - 张量类型: Float32
  - 值范围: [-1.000, 1.000]

✓ 简单模型推理成功!
  - 输出形状: (1, 10)
  - Top-1类别: 3
  - Top-1置信度: 11.96%

✓ 测试完成!
```

### 完整推理输出

```
============================================================
推理视频: 1723591810067-CLEAR.mp4
============================================================
正在加载视频: ../test/1723591810067-CLEAR.mp4
  - 总帧数: 67
  - FPS: 29.92
  - 成功采样 67 帧 (间隔: 1)
输入形状: (1, 3, 67, 224, 224)

正在进行推理...

============================================================
Top-10 预测结果:
============================================================
1. clear                - 置信度: 45.67%
2. clean                - 置信度: 12.34%
3. bright               - 置信度: 8.90%
4. white                - 置信度: 5.67%
5. glass                - 置信度: 4.32%
...

============================================================
推理完成总结
============================================================
总共处理: 6 个视频

所有视频的Top-1预测:
  1723591810067-CLEAR.mp4              -> clear          (45.67%)
  2592423368318-BAT.mp4                -> bat            (38.92%)
  ...

✓ 推理完成!
```

## 🔧 MindSpore vs PyTorch

### 主要差异

| 方面 | PyTorch | MindSpore |
|------|---------|-----------|
| 基类 | `nn.Module` | `nn.Cell` |
| 前向函数 | `forward()` | `construct()` |
| 上下文 | `torch.no_grad()` | `model.set_train(False)` |
| 权重格式 | `.pt`, `.pth` | `.ckpt` |
| 设备设置 | `device = torch.device()` | `context.set_context()` |

### 代码对比示例

**PyTorch:**
```python
model = Model()
model.cuda()
model.eval()
with torch.no_grad():
    output = model(input)
```

**MindSpore:**
```python
from mindspore import context
context.set_context(device_target="CPU")
model = Model()
model.set_train(False)
output = model(input)
```

## 📝 模型架构说明

I3D (Inflated 3D ConvNet) 模型架构：

```
输入: (1, 3, T, 224, 224)
  ↓
Conv3d_1a_7x7 (64通道)
  ↓
MaxPool3d
  ↓
Inception模块 x5
  ↓
AvgPool3d + Dropout
  ↓
Logits (num_classes)
  ↓
输出: (1, num_classes, T')
```

主要特点：
- 3D卷积用于时空特征提取
- Inception模块用于多尺度特征
- 支持可变长度视频输入

## ⚠️ 注意事项

### 1. 内存使用

MindSpore CPU推理内存占用：
- 单个视频（64帧）: ~2-3GB
- 模型加载: ~500MB
- 建议最小RAM: 4GB

### 2. 推理速度

CPU推理速度参考（Intel i7）：
- 单个视频（67帧）: ~10-30秒
- 取决于CPU性能和视频长度

### 3. 权重转换

从PyTorch转换时注意：
- 参数名称需要匹配
- BatchNorm的momentum定义不同（PyTorch: 0.1 → MindSpore: 0.9）
- 某些操作可能需要调整

### 4. 兼容性

- MindSpore 2.0.0 推荐 Python 3.7-3.9
- 需要 glibc >= 2.17
- 不支持Windows原生（需WSL）

## 🐛 故障排查

### 问题1: 导入MindSpore失败

```
ImportError: libgomp.so.1: cannot open shared object file
```

**解决:**
```bash
sudo apt-get install libgomp1
```

### 问题2: GLIBC版本问题

```
version `GLIBC_2.27' not found
```

**解决:** 升级系统或使用Docker

### 问题3: 权重加载失败

```
ValueError: The parameter name ... is not match
```

**解决:** 检查权重文件是否正确转换，参数名称是否匹配

### 问题4: 推理结果不一致

PyTorch和MindSpore结果可能有微小差异（<1%），这是正常的：
- 浮点运算精度差异
- 不同的优化策略
- BatchNorm的实现差异

## 🔗 相关资源

### MindSpore官方资源
- 官网: https://www.mindspore.cn/
- 文档: https://www.mindspore.cn/docs/zh-CN/master/index.html
- API参考: https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html
- 教程: https://www.mindspore.cn/tutorials/zh-CN/master/index.html

### WLASL项目
- 项目主页: https://dxli94.github.io/WLASL/
- GitHub: https://github.com/dxli94/WLASL
- 论文: Word-level Deep Sign Language Recognition from Video (WACV 2020)

## 📚 API快速参考

### 常用MindSpore操作

```python
# 设置运行模式
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 创建Tensor
from mindspore import Tensor
import mindspore.dtype as mstype
x = Tensor([[1, 2], [3, 4]], mstype.float32)

# 加载/保存权重
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
param_dict = load_checkpoint("model.ckpt")
load_param_into_net(net, param_dict)
save_checkpoint(net, "model.ckpt")

# 模型推理
net.set_train(False)
output = net(input_data)
```

## 🎯 进阶使用

### 自定义类别数

修改 `inference_mindspore.py` 中的配置：

```python
NUM_CLASSES = 100  # 改为100, 300, 1000, 或 2000
CLASS_FILE = '../code/I3D/preprocess/wlasl_class_list.txt'
WEIGHTS_PATH = 'weights/i3d_wlasl100.ckpt'  # 对应的权重
```

### 批量推理

修改代码以支持批量输入：

```python
# 加载多个视频
video_tensors = [load_rgb_frames_from_video(v) for v in video_list]
# 拼接成batch
batch = ops.Concat(axis=0)(video_tensors)
# 批量推理
outputs = model(batch)
```

### 使用自己的视频

```bash
# 复制视频到test目录
cp /path/to/your/video.mp4 ../test/

# 运行推理
python inference_mindspore.py
```

## 📊 性能优化建议

1. **使用图模式**: `context.set_context(mode=context.GRAPH_MODE)` （默认）
2. **调整采样率**: 修改 `sample_interval` 减少帧数
3. **减小batch size**: 单个视频推理（已默认）
4. **使用混合精度**: 对于支持的硬件

## 📄 许可证

本代码遵循原WLASL项目的许可证（Computational Use of Data Agreement）。

仅供学术研究使用，禁止商业用途。

---

**版本**: 1.0  
**更新日期**: 2025-10-26  
**框架**: MindSpore 2.0.0  
**设备**: CPU  
**状态**: 测试版
