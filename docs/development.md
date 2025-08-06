# SignAvatar Web 开发指南

## 项目概述

SignAvatar Web 是一个基于 AI 的实时手语识别与 3D 虚拟人播报系统，旨在为听障人群提供高质量的手语翻译服务。

## 技术架构

### 系统架构图

```
浏览器 WebRTC/Canvas → Ascend Edge Gateway → MediaPipe Holistic → 
Transformer-CTC CSLR → 后处理模块 → Avatar Service → WebGL渲染
```

### 核心组件

1. **前端 (React + TypeScript)**
   - 摄像头视频捕获
   - MediaPipe 关键点提取
   - WebSocket 实时通信
   - 3D 虚拟人渲染 (Three.js)
   - 实时字幕显示

2. **后端 (FastAPI + Python)**
   - WebSocket 服务器
   - CSLR 模型推理
   - 性能监控
   - API 接口

3. **AI 模型**
   - MediaPipe Holistic (关键点提取)
   - ST-Transformer-CTC (手语识别)
   - MindSpore Lite (模型推理)

## 开发环境设置

### 前置要求

- Python 3.8+
- Node.js 16+
- Conda 环境管理器
- Git

### 环境配置

1. **克隆项目**
```bash
git clone <repository-url>
cd signavatar-web
```

2. **设置 Python 环境**
```bash
conda create -n shengteng python=3.11
conda activate shengteng
pip install -r requirements.txt
```

3. **设置前端环境**
```bash
cd frontend
npm install
```

4. **环境变量配置**
```bash
cp .env.example .env
# 编辑 .env 文件配置相关参数
```

## 开发流程

### 启动开发服务器

1. **启动后端服务**
```bash
conda activate shengteng
cd backend
python main.py
```

2. **启动前端服务**
```bash
cd frontend
npm run dev
```

3. **访问应用**
- 前端: http://localhost:3001
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/api/docs

### 代码结构

```
SignAvatar-Web/
├── backend/                 # 后端服务
│   ├── api/                # API 路由
│   │   └── websocket.py    # WebSocket 管理
│   ├── models/             # AI 模型
│   ├── services/           # 业务逻辑
│   │   ├── mediapipe_service.py
│   │   └── cslr_service.py
│   ├── utils/              # 工具函数
│   │   ├── config.py       # 配置管理
│   │   └── logger.py       # 日志配置
│   └── main.py             # 主应用
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/     # React 组件
│   │   │   ├── VideoCapture.tsx
│   │   │   ├── SubtitleDisplay.tsx
│   │   │   ├── AvatarViewer.tsx
│   │   │   └── ...
│   │   ├── services/       # API 服务
│   │   │   └── websocket.ts
│   │   ├── hooks/          # React Hooks
│   │   │   └── useSignLanguageRecognition.ts
│   │   └── utils/          # 工具函数
│   └── public/             # 静态资源
├── models/                 # 预训练模型
├── docs/                   # 文档
└── tests/                  # 测试文件
```

## 核心功能开发

### 1. 关键点提取

MediaPipe Holistic 提取 543 个关键点：
- 姿态: 33 个点
- 左手: 21 个点  
- 右手: 21 个点
- 面部: 468 个点

```python
# backend/services/mediapipe_service.py
def extract_landmarks(self, image: np.ndarray) -> Dict:
    # 处理图像并提取关键点
    results = self.holistic.process(rgb_image)
    return self._extract_all_landmarks(results)
```

### 2. 手语识别

使用 ST-Transformer-CTC 模型进行连续手语识别：

```python
# backend/services/cslr_service.py
async def predict(self, landmarks_sequence: List[List[float]]) -> Dict:
    # 准备输入数据
    input_data = self._prepare_input_data()
    
    # 模型推理
    prediction = await self._mindspore_inference(input_data)
    
    # CTC 解码
    decoded_result = self._ctc_decode(prediction)
    
    return self._post_process(decoded_result)
```

### 3. 实时通信

WebSocket 实现低延迟通信：

```typescript
// frontend/src/services/websocket.ts
export class WebSocketService {
  sendLandmarks(landmarks: number[][]): void {
    const message = {
      type: 'landmarks',
      payload: { landmarks, timestamp: Date.now() }
    }
    this.send(message)
  }
}
```

### 4. 3D 虚拟人

使用 Three.js 渲染 3D 虚拟人：

```typescript
// frontend/src/components/ThreeAvatar.tsx
const AvatarMesh: React.FC = ({ text, isActive, animationType }) => {
  useFrame((state, delta) => {
    // 根据动画类型更新虚拟人姿态
    switch (animationType) {
      case '挥手':
        meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 3) * 0.1
        break
      // ...
    }
  })
}
```

## 性能优化

### 1. 延迟优化

- 限制 WebSocket 发送频率 (30fps)
- 关键点数据压缩
- 批处理优化
- 模型量化 (INT8)

### 2. 内存优化

- 序列缓冲区管理
- 资源及时释放
- 图像处理优化

### 3. 网络优化

- WebSocket 连接池
- 数据压缩
- 错误重试机制

## 测试

### 单元测试

```bash
# 后端测试
cd backend
python -m pytest tests/

# 前端测试
cd frontend
npm test
```

### 集成测试

```bash
# 系统测试
python test_system.py
```

### 性能测试

- 端到端延迟 ≤ 150ms
- 帧率 ≥ 30fps
- 内存使用 < 1GB

## 部署

### Docker 部署

```bash
# 构建和启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 生产环境

1. **环境变量配置**
```bash
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=your-production-secret-key
```

2. **性能监控**
- Prometheus 指标收集
- Grafana 仪表板
- 日志聚合

3. **安全配置**
- HTTPS 证书
- CORS 策略
- 访问控制

## 故障排除

### 常见问题

1. **摄像头无法访问**
   - 检查浏览器权限
   - 确认设备可用性

2. **WebSocket 连接失败**
   - 检查后端服务状态
   - 验证网络连接

3. **识别准确率低**
   - 调整置信度阈值
   - 改善光线条件
   - 标准化手语动作

4. **性能问题**
   - 关闭关键点显示
   - 降低视频质量
   - 检查系统资源

### 调试工具

- 浏览器开发者工具
- 性能监控面板
- 后端日志
- WebSocket 消息追踪

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 代码规范

- Python: PEP 8
- TypeScript: ESLint + Prettier
- 提交信息: Conventional Commits

## 许可证

MIT License
