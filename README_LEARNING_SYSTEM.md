# 🎓 手语学习训练系统

专业的手语学习平台，为学习者提供系统化的手语教学功能。

## 🌟 核心功能

### 📚 学习系统
- **系统化学习路径** - 从基础到高级的完整学习体系
- **互动式手语练习** - 实时手语识别和反馈
- **实时进度跟踪** - 详细的学习进度和成就系统
- **成就系统激励** - 丰富的成就和奖励机制
- **个性化推荐** - 基于学习情况的智能推荐

### 🔧 技术特色
- **现代化界面** - React + TypeScript + Material-UI
- **高性能后端** - Python FastAPI + SQLite
- **实时通信** - WebSocket 支持
- **响应式设计** - 适配各种设备

## 🚀 快速开始

### 方式一：一键启动（推荐）
```bash
# Windows用户
start_learning_system.bat

# Linux/Mac用户
python quick_start_learning.py
```

### 方式二：手动启动

#### 1. 启动后端服务
```bash
cd backend
python main.py
```

#### 2. 启动前端服务
```bash
cd frontend
npm install  # 首次运行
npm run dev
```

#### 3. 访问系统
- 🎓 学习平台: http://localhost:5173/learning
- 📚 API文档: http://localhost:8000/docs
- 💓 系统状态: http://localhost:8000/health

## 📖 学习内容

### 基础学习模块
1. **问候与介绍**
   - 你好、再见
   - 自我介绍
   - 基本礼貌用语

2. **数字与时间**
   - 0-100数字表达
   - 时间概念
   - 日期表达

3. **家庭关系**
   - 家庭成员称谓
   - 人际关系表达
   - 情感表达

4. **日常生活**
   - 衣食住行
   - 工作学习
   - 兴趣爱好

5. **高级表达**
   - 复杂语法结构
   - 专业词汇
   - 情景对话

### 学习特色

#### 🎯 个性化学习路径
- 根据学习进度自动调整难度
- 智能推荐相关学习内容
- 个性化复习计划

#### 🏆 成就系统
- 学习里程碑奖励
- 连续学习天数记录
- 技能掌握认证

#### 📊 详细统计
- 学习时间统计
- 进度可视化
- 知识点掌握情况

## 🛠️ 系统架构

```
手语学习训练系统/
├── backend/                 # 后端服务
│   ├── main.py             # 主应用入口
│   ├── services/           # 业务服务层
│   │   └── learning_training_service.py
│   └── api/                # API路由层
│       └── learning_routes.py
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── pages/
│   │   │   └── LearningPage.tsx
│   │   └── components/
│   └── package.json
└── data/                   # 数据存储
    └── learning.db        # SQLite数据库
```

## 🔧 技术栈

### 后端技术
- **Python 3.8+** - 核心语言
- **FastAPI** - Web框架
- **SQLite** - 数据库
- **Uvicorn** - ASGI服务器
- **WebSockets** - 实时通信

### 前端技术
- **React 18** - UI框架
- **TypeScript** - 类型安全
- **Material-UI** - 组件库
- **Vite** - 构建工具

## 📋 环境要求

### 基础环境
- Python 3.8+ 
- Node.js 16+
- npm 8+

### Python依赖
```bash
pip install fastapi uvicorn websockets aiosqlite
```

### 前端依赖
```bash
npm install
```

## 🎮 使用指南

### 学习者界面

1. **选择学习模块**
   - 浏览可用的学习模块
   - 查看学习要求和目标
   - 开始新的学习课程

2. **完成学习任务**
   - 观看手语视频教程
   - 参与互动练习
   - 完成课程测试

3. **跟踪学习进度**
   - 查看学习统计
   - 监控技能提升
   - 获得成就奖励

### 管理员功能

1. **内容管理**
   - 添加新的学习模块
   - 更新课程内容
   - 管理学习资源

2. **用户管理**
   - 查看学习者进度
   - 分析学习数据
   - 提供个性化建议

## 🔍 API接口

### 学习管理接口
```
GET    /api/learning/modules          # 获取学习模块
GET    /api/learning/progress/{user}  # 获取学习进度
POST   /api/learning/complete         # 完成学习任务
GET    /api/learning/achievements     # 获取成就列表
```

### 系统接口
```
GET    /api/health                    # 系统健康检查
GET    /api/status                    # 服务状态
GET    /docs                          # API文档
```

## 🐛 常见问题

### 启动问题

**Q: 后端服务启动失败？**
A: 检查Python版本和依赖安装，确保端口8000未被占用

**Q: 前端页面无法访问？**
A: 确保Node.js环境正确，运行`npm install`安装依赖

**Q: 数据库连接失败？**
A: 检查data目录权限，确保SQLite文件可读写

### 功能问题

**Q: 学习进度没有保存？**
A: 检查后端API连接，确保网络通畅

**Q: 视频无法播放？**
A: 检查浏览器设置，允许媒体自动播放

## 🤝 贡献指南

欢迎为手语学习训练系统贡献代码！

1. Fork 项目
2. 创建功能分支
3. 提交代码更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系我们

- 项目维护者: 手语学习团队
- 问题反馈: 通过GitHub Issues
- 技术支持: 查看在线文档

---

🎓 **让学习手语变得更简单、更有趣！**
