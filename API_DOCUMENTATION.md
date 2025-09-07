# 手语学习训练系统 API 文档

## 概述

本文档描述了手语学习训练系统的 REST API 和 WebSocket API。

- **基础URL**: `http://localhost:8000`
- **API版本**: v1
- **认证方式**: JWT Token
- **数据格式**: JSON

## 认证

### 获取访问令牌

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600
  }
}
```

### 使用令牌

在请求头中包含访问令牌：
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## REST API

### 系统管理

#### 获取系统健康状态
```http
GET /api/system/health
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "overall_status": "healthy",
    "services": [
      {
        "name": "database",
        "status": "running",
        "uptime": 3600.5
      }
    ],
    "performance": {
      "cpu_usage": 45.2,
      "memory_usage": 68.5
    }
  }
}
```

#### 获取系统统计
```http
GET /api/system/services
```

#### 重启服务
```http
POST /api/system/services/{service_name}/restart
```

### 文件管理

#### 上传文件
```http
POST /api/files/upload
Content-Type: multipart/form-data

file: [binary data]
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "file_id": "uuid-string",
    "filename": "video.mp4",
    "file_size": 1024000,
    "file_path": "/uploads/video.mp4"
  }
}
```

#### 获取文件信息
```http
GET /api/files/{file_id}
```

#### 删除文件
```http
DELETE /api/files/{file_id}
```

### 手语识别

#### 识别视频文件
```http
POST /api/recognition/video
Content-Type: application/json

{
  "file_id": "uuid-string",
  "config": {
    "confidence_threshold": 0.7,
    "max_sequence_length": 100
  }
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "task_id": "task-uuid",
    "status": "pending"
  }
}
```

#### 获取识别结果
```http
GET /api/recognition/tasks/{task_id}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "task_id": "task-uuid",
    "status": "completed",
    "result": {
      "text": "你好世界",
      "confidence": 0.95,
      "gloss_sequence": ["你好", "世界"],
      "processing_time": 2.5
    }
  }
}
```

### 学习训练

#### 获取课程列表
```http
GET /api/learning/courses?page=1&page_size=20
```

**响应**:
```json
{
  "status": "success",
  "data": [
    {
      "course_id": "course-1",
      "title": "基础手语入门",
      "description": "从零开始学习手语",
      "difficulty_level": 1,
      "estimated_hours": 20.0,
      "progress": 0.3
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 50,
    "total_pages": 3
  }
}
```

#### 注册课程
```http
POST /api/learning/courses/{course_id}/enroll
```

#### 获取课程详情
```http
GET /api/learning/courses/{course_id}
```

#### 获取课时列表
```http
GET /api/learning/courses/{course_id}/lessons
```

#### 完成课时
```http
POST /api/learning/lessons/{lesson_id}/complete
Content-Type: application/json

{
  "score": 85,
  "time_spent": 1800,
  "answers": [...]
}
```

### 成就系统

#### 获取用户成就
```http
GET /api/achievements?include_locked=true
```

**响应**:
```json
{
  "status": "success",
  "data": [
    {
      "achievement_id": "first_lesson",
      "title": "初学者",
      "description": "完成第一个课程",
      "category": "learning",
      "rarity": "common",
      "icon": "🎓",
      "xp_reward": 50,
      "is_unlocked": true,
      "unlocked_at": "2023-12-01T10:00:00Z",
      "progress": 1.0
    }
  ]
}
```

#### 触发成就检查
```http
POST /api/achievements/check
Content-Type: application/json

{
  "event_type": "lesson_completed",
  "event_data": {
    "lesson_id": "lesson-1",
    "score": 95
  }
}
```

## WebSocket API

### 连接端点

#### 实时手语识别
```
ws://localhost:8000/ws/sign-recognition
```

#### 学习训练
```
ws://localhost:8000/ws/learning
```

#### 系统监控
```
ws://localhost:8000/ws/system
```

### 消息格式

所有WebSocket消息都使用以下格式：

```json
{
  "type": "message_type",
  "payload": {
    "key": "value"
  },
  "timestamp": 1701234567.89,
  "message_id": "uuid-string"
}
```

### 实时手语识别

#### 开始识别会话
```json
{
  "type": "start_recognition",
  "payload": {
    "session_id": "session-1",
    "config": {
      "min_frames": 8,
      "confidence_threshold": 0.7
    }
  }
}
```

#### 发送关键点数据
```json
{
  "type": "landmarks",
  "payload": {
    "landmarks": [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6]
    ],
    "timestamp": 1701234567.89,
    "frame_id": 123
  }
}
```

#### 接收识别结果
```json
{
  "type": "recognition_result",
  "payload": {
    "text": "你好",
    "confidence": 0.95,
    "gloss_sequence": ["你好"],
    "frame_id": 123,
    "processing_time": 0.1
  }
}
```

#### 停止识别会话
```json
{
  "type": "stop_recognition",
  "payload": {
    "session_id": "session-1"
  }
}
```

### 学习训练

#### 开始学习会话
```json
{
  "type": "start_session",
  "payload": {
    "course_id": "course-1",
    "lesson_id": "lesson-1"
  }
}
```

#### 更新学习进度
```json
{
  "type": "lesson_progress",
  "payload": {
    "lesson_id": "lesson-1",
    "progress": 0.5,
    "score": 85
  }
}
```

#### 提交练习结果
```json
{
  "type": "exercise_result",
  "payload": {
    "exercise_id": "exercise-1",
    "score": 90,
    "accuracy": 0.95,
    "time_spent": 300,
    "mistakes": []
  }
}
```

### 系统监控

#### 获取系统统计
```json
{
  "type": "get_system_stats",
  "payload": {}
}
```

#### 接收统计数据
```json
{
  "type": "system_stats",
  "payload": {
    "websocket": {
      "active_connections": 10,
      "total_messages": 1000
    },
    "recognition": {
      "active_sessions": 3,
      "total_recognitions": 500
    }
  }
}
```

## 错误处理

### HTTP状态码

- `200 OK`: 请求成功
- `201 Created`: 资源创建成功
- `400 Bad Request`: 请求参数错误
- `401 Unauthorized`: 未授权
- `403 Forbidden`: 禁止访问
- `404 Not Found`: 资源不存在
- `422 Unprocessable Entity`: 数据验证失败
- `500 Internal Server Error`: 服务器内部错误

### 错误响应格式

```json
{
  "status": "error",
  "message": "错误描述",
  "error": {
    "code": "ERROR_CODE",
    "message": "详细错误信息",
    "field": "字段名",
    "details": {}
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

### 常见错误代码

- `VALIDATION_ERROR`: 数据验证失败
- `AUTHENTICATION_FAILED`: 认证失败
- `PERMISSION_DENIED`: 权限不足
- `RESOURCE_NOT_FOUND`: 资源不存在
- `SERVICE_UNAVAILABLE`: 服务不可用
- `RECOGNITION_FAILED`: 识别失败
- `FILE_TOO_LARGE`: 文件过大

## 限制和配额

### 请求限制
- **API请求**: 1000次/小时/用户
- **文件上传**: 100MB/文件
- **WebSocket连接**: 5个/用户

### 数据限制
- **识别视频**: 最长10分钟
- **批量操作**: 最多100项/请求
- **消息大小**: 最大1MB

## SDK和示例

### JavaScript SDK
```javascript
import { SignLanguageAPI } from 'sign-language-sdk';

const api = new SignLanguageAPI({
  baseURL: 'http://localhost:8000',
  token: 'your-access-token'
});

// 上传并识别视频
const result = await api.recognition.recognizeVideo(file);
console.log(result.text);
```

### Python SDK
```python
from sign_language_sdk import SignLanguageAPI

api = SignLanguageAPI(
    base_url='http://localhost:8000',
    token='your-access-token'
)

# 获取课程列表
courses = api.learning.get_courses()
print(courses)
```

## 更新日志

### v1.0.0 (2023-12-01)
- 初始版本发布
- 基础手语识别功能
- 学习训练系统
- 成就系统

### v1.1.0 (计划中)
- 实时协作功能
- 高级分析报告
- 多语言支持

## 技术支持

如需技术支持，请：

1. 查看 [FAQ](FAQ.md)
2. 提交 [Issue](https://github.com/your-org/sign-language-learning/issues)
3. 联系技术支持: support@example.com
