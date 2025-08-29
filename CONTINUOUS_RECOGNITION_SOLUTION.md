# 手语识别系统说明

## 问题说明

您遇到的问题是：**Web应用目前使用的是模拟的简化服务，只能识别单个词汇，而不是真正的连续手语识别**。

## 两种识别方式的区别

### 1. 增强版CE-CSL服务 (当前默认)
- **路径**: `/api/enhanced-cecsl/upload-video`
- **实现**: `SimpleEnhancedCECSLService` (模拟服务)
- **输出**: 随机选择1-3个单个词汇
- **用途**: 演示和测试用，非真实识别

### 2. 连续手语识别服务 (新增的真实服务)
- **路径**: `/api/sign-recognition/upload-video`
- **实现**: `SignRecognitionService` + `CSLRService` + `MediaPipeService`
- **处理流程**:
  1. 视频抽帧 (25fps)
  2. MediaPipe关键点提取 (543个关键点)
  3. 滑动窗口分割
  4. CSLR模型推理
  5. CTC解码为gloss序列
  6. 规则翻译为连续句子
- **输出**: 完整的连续手语句子

## 解决方案

我已经为您完成以下修改：

### 后端改动

1. **添加真实识别服务导入** (`backend/main.py`)
   ```python
   from services.sign_recognition_service import SignRecognitionService
   from services.mediapipe_service import MediaPipeService
   from services.cslr_service import CSLRService
   ```

2. **初始化连续手语识别服务**
   ```python
   mediapipe_service = MediaPipeService()
   cslr_service = CSLRService()
   sign_recognition_service = SignRecognitionService(mediapipe_service, cslr_service)
   ```

3. **新增API端点**
   - `POST /api/sign-recognition/upload-video` - 上传视频进行连续识别
   - `GET /api/sign-recognition/status/{task_id}` - 查询识别状态

### 前端改动

1. **新增服务类** (`frontend/src/services/continuousSignRecognitionService.ts`)
   - 调用真正的CSLR API
   - 支持进度追踪和结果轮询

2. **新增组件** (`frontend/src/components/ContinuousVideoRecognition.tsx`)
   - 专门用于连续手语识别的界面
   - 显示完整的gloss序列和分段信息

3. **更新识别页面** (`frontend/src/pages/RecognitionPage.tsx`)
   - 同时提供两种识别方式对比

## 使用方法

1. **启动后端服务**
   ```bash
   cd backend
   python main.py
   ```

2. **启动前端**
   ```bash
   cd frontend
   npm start
   ```

3. **访问识别页面**
   - 打开浏览器访问 `http://localhost:3000`
   - 进入"识别页面"
   - 现在您将看到两个视频上传组件：
     - "增强版CE-CSL服务" (模拟，单词识别)
     - "连续手语识别" (真实，句子识别)

## 识别效果对比

### 模拟服务输出示例：
```json
{
  "text": "你好",
  "confidence": 0.85,
  "gloss_sequence": ["你好", "谢谢"]
}
```

### 连续识别输出示例：
```json
{
  "text": "你好，我想学习手语。谢谢！",
  "gloss_sequence": ["你好", "我", "想", "学习", "手语", "谢谢"],
  "segments": [
    {
      "gloss_sequence": ["你好"],
      "start_time": 0.0,
      "end_time": 1.2,
      "confidence": 0.92
    },
    {
      "gloss_sequence": ["我", "想", "学习", "手语"],
      "start_time": 1.2,
      "end_time": 4.8,
      "confidence": 0.87
    }
  ]
}
```

## 注意事项

1. **模型依赖**: 连续识别需要加载训练好的CSLR模型文件
2. **性能要求**: 真实识别比模拟服务耗时更长
3. **内存占用**: MediaPipe和CSLR模型需要较多内存

现在您可以使用真正的连续手语识别功能了！
