# MindSpore版本的VAC_CSLR推理 + LLM句子生成

这个文件夹包含了使用MindSpore框架实现的手语识别推理代码,并集成了通义千问大语言模型,可以将识别出的零散词汇转换为流畅的中英文完整句子。

## ✨ 主要特性

- 🎥 **手语视频识别**: 基于MindSpore的高性能手语识别推理
- 🤖 **LLM句子生成**: 通过通义千问API将零散词汇转换为完整句子
- 🌐 **中英对译**: 自动生成中文和英文两个版本
- 📊 **多格式输出**: 同时保存TXT和JSON格式的结果

## 📁 文件结构

```
mind_vac/
├── model.py                # MindSpore模型定义
├── decoder.py              # CTC解码器
├── transforms.py           # 数据预处理和增强
├── inference.py            # 推理主程序(已集成LLM)
├── qwen_api.py            # 通义千问API集成
├── config.py              # 配置文件
├── convert_weights.py      # PyTorch权重转MindSpore工具
├── run_inference.sh       # 快速推理脚本
├── setup_api.sh           # API配置助手
├── .env.example           # 环境变量示例
├── README.md              # 本文档
└── LLM_INTEGRATION.md     # LLM集成详细文档
```

## � 快速开始

### 方法1: 使用快速脚本(推荐)

```bash
# 1. 激活conda环境
conda activate vac_cslr

# 2. 配置API密钥(首次使用)
./setup_api.sh

# 3. 运行推理(基本模式)
./run_inference.sh test/1

# 4. 运行推理(使用LLM生成完整句子)
./run_inference.sh test/1 --use-llm
```

### 方法2: 手动命令

#### 步骤1: 安装依赖

```bash
# 激活conda环境
conda activate vac_cslr

# 安装requests库(用于API调用)
pip install requests
```

#### 步骤2: 配置通义千问API(可选,启用LLM功能时需要)

```bash
# 方式1: 设置环境变量(推荐)
export DASHSCOPE_API_KEY="your_api_key_here"

# 方式2: 使用配置助手
./setup_api.sh

# 方式3: 创建.env文件
cp .env.example .env
# 然后编辑.env文件,填入你的API密钥
```

获取API密钥: https://dashscope.console.aliyun.com/apiKey

#### 步骤3: 运行推理

**基本推理(仅识别):**
```bash
python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --device CPU \
    --output ./output_dir
```

**完整推理(使用LLM生成句子):**
```bash
python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --device CPU \
    --output ./output_dir \
    --use-llm
```

## 📊 输出示例

### 基本模式(仅识别)

```
Using device: CPU
MindSpore version: 2.0.0

Loading gloss dictionary with 1295 entries
Creating model...
Loading checkpoint from slr_mindspore.ckpt
Model loaded successfully!

Loading video from: test/1
Found 176 frames
Preprocessing video...
Running inference...

============================================================
Inference Result:
============================================================
Video: 1
Frames: 176
Recognized Gloss: __ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA
============================================================

Results saved to:
  - ./output_dir/inference_result.txt
  - ./output_dir/inference_result.json
```

### LLM模式(生成完整句子)

```
Using device: CPU
MindSpore version: 2.0.0

[... 模型加载和推理过程 ...]

============================================================
Inference Result:
============================================================
Video: 1
Frames: 176
Recognized Gloss: __ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA

------------------------------------------------------------
Calling Qwen API to generate complete sentences...
------------------------------------------------------------

完整句子翻译:
中文: 亲爱的观众,晚上好。冬季该地区在美国发生了洪水。
English: Dear viewers, good evening. Floods occurred in the region in America during winter.
置信度: high
说明: 识别到了问候语和关于美国某地区冬季洪水的新闻内容
============================================================

Results saved to:
  - ./output_dir/inference_result.txt
  - ./output_dir/inference_result.json
```

### 输出文件格式

**inference_result.txt** - 文本格式:
```
Video: test/1
Frames: 176
Recognized Gloss: __ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA

============================================================
LLM Generated Complete Sentences:
============================================================
中文: 亲爱的观众,晚上好。冬季该地区在美国发生了洪水。
English: Dear viewers, good evening. Floods occurred in the region in America during winter.
置信度: high
说明: 识别到了问候语和关于美国某地区冬季洪水的新闻内容
```

**inference_result.json** - JSON格式:
```json
{
  "video_path": "test/1",
  "frames": 176,
  "recognized_gloss": "__ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA",
  "gloss_words": ["__ON__", "LIEB", "ZUSCHAUER", "ABEND", "WINTER", "NULL", "loc-REGION", "UEBERSCHWEMMUNG", "AMERIKA"],
  "llm_result": {
    "chinese": "亲爱的观众,晚上好。冬季该地区在美国发生了洪水。",
    "english": "Dear viewers, good evening. Floods occurred in the region in America during winter.",
    "confidence": "high",
    "explanation": "识别到了问候语和关于美国某地区冬季洪水的新闻内容",
    "success": true
  }
}
```

## 🎯 命令行参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--video-path` | str | 是 | - | 视频帧文件夹路径 |
| `--checkpoint` | str | 是 | - | MindSpore模型权重路径 |
| `--dict-path` | str | 否 | gloss_dict.npy | 手语词汇字典路径 |
| `--device` | str | 否 | CPU | 运行设备(CPU/GPU/Ascend) |
| `--output` | str | 否 | ./mind_inference_output | 输出目录 |
| `--use-llm` | flag | 否 | False | 是否使用LLM生成完整句子 |
| `--api-key` | str | 否 | None | 通义千问API密钥 |
| `--qwen-model` | str | 否 | qwen-plus | 使用的通义千问模型 |

## 🤖 通义千问模型选择

| 模型 | 特点 | 适用场景 | 相对成本 |
|------|------|----------|----------|
| `qwen-turbo` | 速度快 | 大批量处理 | 低 |
| `qwen-plus` | 平衡性能(推荐) | 日常使用 | 中 |
| `qwen-max` | 最强性能 | 高质量要求 | 高 |

使用示例:
```bash
python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --use-llm \
    --qwen-model qwen-max
```

## 📚 详细文档

- **LLM集成详细说明**: 查看 [LLM_INTEGRATION.md](LLM_INTEGRATION.md)
- **API配置指南**: 运行 `./setup_api.sh` 或查看 `.env.example`
- **MindSpore迁移**: 查看本文档后续章节

## 🔧 环境安装(首次设置)

### 1. 安装MindSpore

```bash
# CPU版本
pip install mindspore

# GPU版本 (CUDA 11.1)
pip install mindspore-gpu

# Ascend版本
pip install mindspore-ascend
```

### 2. 安装其他依赖

```bash
pip install opencv-python numpy requests
```

### 3. 转换PyTorch权重到MindSpore格式(如果需要)

```bash
python convert_weights.py \
    --pytorch-ckpt resnet18__slr_pretrained.pt \
    --output slr_mindspore.ckpt
```

## 🔄 PyTorch vs MindSpore 主要差异

| 组件 | PyTorch | MindSpore |
|-----|---------|-----------|
| 模型基类 | `nn.Module` | `nn.Cell` |
| 前向函数 | `forward()` | `construct()` |
| 张量 | `torch.Tensor` | `ms.Tensor` |
| 参数 | `nn.Parameter` | `ms.Parameter` |
| 数据维度 | `(B, C, H, W)` | `(B, C, H, W)` (相同) |

## ⚠️ 注意事项

1. **API密钥安全**: 请妥善保管API密钥,不要提交到版本控制系统

2. **API费用**: 通义千问API按调用量计费,请注意使用成本

3. **网络连接**: 使用LLM功能需要稳定的网络连接

4. **权重转换**: PyTorch和MindSpore的权重格式不同,需要使用`convert_weights.py`进行转换

5. **LSTM参数**: LSTM的权重组织方式在两个框架中有所不同,可能需要手动调整

6. **BatchNorm参数**: 
   - PyTorch: `running_mean`, `running_var`
   - MindSpore: `moving_mean`, `moving_variance`

## 🚀 性能优化建议

1. **静态图模式**: 使用`context.GRAPH_MODE`可以获得更好的性能
2. **混合精度**: 在GPU/Ascend上可以使用FP16加速
3. **批处理**: 处理多个视频时可以使用批处理提高效率
4. **模型选择**: 根据需求选择合适的通义千问模型(turbo/plus/max)

## 📝 批处理示例

```bash
#!/bin/bash
# 批量处理多个视频

for video in test/*; do
    echo "Processing: $video"
    python inference.py \
        --video-path "$video" \
        --checkpoint slr_mindspore.ckpt \
        --dict-path gloss_dict.npy \
        --use-llm \
        --output "./output_dir/$(basename $video)"
done
```

## ❓ 常见问题

### Q1: 如何获取通义千问API密钥?
A: 访问 https://dashscope.console.aliyun.com/apiKey ,登录/注册阿里云账号,创建并复制API密钥

### Q2: API调用失败怎么办?
A: 检查以下几点:
- API密钥是否正确设置
- 网络连接是否正常
- 阿里云账户是否有余额
- 是否被API频率限制

### Q3: LLM翻译结果不理想?
A: 可以尝试:
- 更换更强大的模型(如qwen-max)
- 在`qwen_api.py`中调整提示词
- 检查手语识别结果是否准确

### Q4: 如何不使用LLM功能?
A: 简单!去掉`--use-llm`参数即可,程序会只进行手语识别

### Q5: 转换权重时出错怎么办?
A: 检查PyTorch checkpoint文件是否完整,确保包含`model_state_dict`字段

### Q6: 推理结果与PyTorch版本不一致?
A: 可能是权重转换或数值精度问题,检查模型参数是否正确加载

### Q7: 如何在Ascend设备上运行?
A: 设置`--device Ascend`,并确保已安装MindSpore Ascend版本

### Q8: 支持哪些MindSpore版本?
A: 建议使用MindSpore 1.8+或2.0+版本

## 🎉 更新日志

### v1.1.0 (2025-10-25)
- ✨ 新增通义千问API集成
- ✨ 实现零散词汇到完整句子的转换
- ✨ 支持中英文对译输出
- ✨ 添加JSON格式输出
- 🔧 新增快速推理脚本 `run_inference.sh`
- 🔧 新增API配置助手 `setup_api.sh`
- 📝 完善文档和使用说明

### v1.0.0
- 🎉 基础手语识别功能
- 🔄 PyTorch到MindSpore权重转换
- 📊 CTC解码器实现

## 📚 参考资料

- [MindSpore官方文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [MindSpore模型迁移指南](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/overview.html)
- [通义千问API文档](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)
- [原始PyTorch实现](../)

## 🤝 贡献与支持

如果你在使用过程中遇到问题或有改进建议,欢迎提Issue或Pull Request!

## 📄 许可证

本项目遵循原项目的许可证。

---

**祝使用愉快! 🎊**
