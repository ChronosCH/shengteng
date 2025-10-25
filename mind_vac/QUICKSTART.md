# 🚀 5分钟快速入门指南

本指南帮助你快速开始使用手语识别 + LLM句子生成功能。

## 第一步: 准备环境 (1分钟)

```bash
# 激活conda环境
conda activate vac_cslr

# 安装requests(如果还没安装)
pip install requests
```

## 第二步: 配置API密钥 (2分钟)

### 方法A: 使用配置助手(推荐)

```bash
./setup_api.sh
```

按照提示操作即可。

### 方法B: 手动配置

1. 访问 https://dashscope.console.aliyun.com/apiKey
2. 登录并创建API密钥
3. 设置环境变量:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 第三步: 运行推理 (2分钟)

### 基本推理(仅识别)

```bash
python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --device CPU \
    --output ./output_dir
```

### 完整推理(使用LLM)

```bash
python inference.py \
    --video-path test/1 \
    --checkpoint slr_mindspore.ckpt \
    --dict-path gloss_dict.npy \
    --device CPU \
    --output ./output_dir \
    --use-llm
```

或者使用快捷脚本:

```bash
./run_inference.sh test/1 --use-llm
```

## 第四步: 查看结果

推理完成后,查看输出目录:

```bash
# 查看文本结果
cat output_dir/inference_result.txt

# 查看JSON结果
cat output_dir/inference_result.json
```

## 预期输出示例

```
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
```

## 🎉 成功!

现在你已经成功运行了手语识别并使用LLM生成了完整句子!

## 下一步

- 📖 阅读完整文档: [README.md](README.md)
- 📖 了解LLM集成细节: [LLM_INTEGRATION.md](LLM_INTEGRATION.md)
- 🧪 查看使用示例: `python example_usage.py`
- 🔧 尝试不同的视频: 将 `test/1` 替换为 `test/2`
- 🤖 尝试不同的模型: 添加 `--qwen-model qwen-max`

## 常见问题

**Q: 没有API密钥怎么办?**
A: 可以先不使用 `--use-llm` 参数,仅进行手语识别。

**Q: API调用失败?**
A: 检查网络连接和API密钥是否正确。

**Q: 结果不理想?**
A: 可以尝试更强大的模型,如 `--qwen-model qwen-max`。

## 获取帮助

```bash
# 查看所有参数
python inference.py --help

# 运行示例代码
python example_usage.py

# 测试API连接
python qwen_api.py
```

---

**祝使用愉快! 🎊**
