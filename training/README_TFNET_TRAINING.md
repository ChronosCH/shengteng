# TFNet连续手语识别训练系统

## 问题解决

本项目解决了原始训练代码中训练和验证准确率始终为0%的问题。

### 主要问题和解决方案

1. **数据加载问题**
   - 原问题：数据路径不匹配，CSV列索引错误
   - 解决方案：修正数据路径，正确解析CE-CSL数据格式

2. **模型架构问题**
   - 原问题：MindSpore转换不完整，缺少关键组件
   - 解决方案：基于PyTorch参考实现完整重构TFNet模型

3. **损失计算问题**
   - 原问题：CTC损失函数配置错误
   - 解决方案：按照参考实现正确配置损失函数和知识蒸馏

4. **批处理问题**
   - 原问题：批处理函数与模型输入不匹配
   - 解决方案：实现与PyTorch版本完全一致的collate_fn

## 文件结构

### 新的训练文件（已修正）
- `tfnet_mindspore_corrected.py` - 修正的TFNet模型实现
- `tfnet_dataset_corrected.py` - 修正的数据集加载器
- `tfnet_trainer_corrected.py` - 修正的训练器
- `train_tfnet.py` - 主训练脚本

### 测试文件
- `quick_test.py` - 数据加载测试
- `test_training.py` - 完整训练测试

### 已删除的旧文件
- `tfnet_trainer.py` - 有问题的旧训练器
- `tfnet_mindspore_new.py` - 有问题的旧模型

## 使用方法

### 方法1：使用批处理文件（推荐）
```bash
# 在项目根目录运行
start_training.bat
```

### 方法2：手动运行
```bash
# 激活conda环境
conda activate shengteng

# 进入训练目录
cd training

# 运行训练
python train_tfnet.py
```

## 配置说明

训练配置已优化为CPU运行：
- 隐藏层大小：512（降低以适应CPU）
- 批次大小：1（减少内存使用）
- 学习率：0.0001
- 设备：CPU

## 数据要求

确保以下数据文件存在：
- `data/CE-CSL/train.corpus.csv`
- `data/CE-CSL/dev.corpus.csv`
- `data/CE-CSL/test.corpus.csv`
- `data/CE-CSL/processed/train/*.npz`
- `data/CE-CSL/processed/dev/*.npz`
- `data/CE-CSL/processed/test/*.npz`

## 预期结果

修正后的训练系统应该能够：
1. 正确加载CE-CSL数据集
2. 构建包含3515个词汇的词汇表
3. 成功进行前向传播
4. 计算有效的CTC损失
5. 实现非零的训练和验证准确率

## 技术细节

### 模型架构
- ResNet34MAM特征提取器
- 双路径时序建模（原始+FFT）
- BiLSTM时序编码器
- 多分类器集成
- 知识蒸馏损失

### 关键改进
1. 完整的PyTorch到MindSpore转换
2. 正确的数据预处理管道
3. 适配CPU的模型配置
4. 有效的损失函数实现

## 故障排除

如果遇到问题：
1. 确保激活了正确的conda环境（shengteng）
2. 检查MindSpore是否正确安装
3. 验证数据文件是否存在
4. 查看训练日志中的错误信息

## 性能优化

为了在CPU上获得更好的性能：
- 减小批次大小
- 降低模型复杂度
- 使用数据并行（如果有多核CPU）
- 考虑使用混合精度训练
