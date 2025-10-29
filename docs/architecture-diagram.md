# SignAvatar 系统架构图

## 整体架构概览

```mermaid
graph TB
    subgraph "SignAvatar - 让沟通无障碍"
        Logo["🐿️ SignAvatar<br/>手语识别与学习一体化平台"]
    end
    
    subgraph "核心定位"
        Position["🎯 基于 FastAPI + MindSpore<br/>的手语识别与学习一体化平台"]
    end
    
    subgraph "三大核心能力"
        Capability1["📹 连续手语识别<br/>Continuous SLR"]
        Capability2["✋ 孤立手语识别<br/>Isolated SLR"]
        Capability3["📚 系统化学习路径<br/>Learning Platform"]
    end
    
    subgraph "技术栈"
        Tech1["Python 3.8+"]
        Tech2["FastAPI"]
        Tech3["MindSpore 2.x"]
        Tech4["React 18"]
        Tech5["WebSocket"]
        Tech6["SQLite/Redis"]
    end
    
    subgraph "开源许可"
        License["⭐ MIT 许可证<br/>开源友好"]
    end
    
    Logo --> Position
    Position --> Capability1
    Position --> Capability2
    Position --> Capability3
    
    Capability1 --> Tech1
    Capability2 --> Tech2
    Capability3 --> Tech3
    Tech1 --> Tech4
    Tech2 --> Tech5
    Tech3 --> Tech6
    
    Tech6 --> License
    
    style Logo fill:#FFB6C1,stroke:#FF69B4,stroke-width:3px
    style Position fill:#FFF0F5,stroke:#FF69B4,stroke-width:2px
    style Capability1 fill:#E6F3FF,stroke:#4A90E2,stroke-width:2px
    style Capability2 fill:#FFF9E6,stroke:#FFB84D,stroke-width:2px
    style Capability3 fill:#E6F7FF,stroke:#52C41A,stroke-width:2px
    style License fill:#F0F5FF,stroke:#597EF7,stroke-width:2px
```

## 核心能力展开图

```mermaid
mindmap
  root((SignAvatar))
    连续手语识别
      Mind-VAC CSLR引擎
      实时WebSocket推理
      批量视频处理
      通义千问LLM翻译
      SRT字幕生成
    孤立手语识别
      I3D模型
      Top-K预测
      mind_wl推理引擎
      短视频分类
      练习反馈
    系统化学习
      课程模块管理
      进度跟踪
      成就系统
      个性化推荐
      游戏化设计
    技术栈
      FastAPI异步框架
      MindSpore AI引擎
      React前端
      WebSocket实时通信
      SQLite/Redis存储
```

## 技术架构三层图

```mermaid
graph LR
    subgraph "前端层 Presentation"
        A1[React 18 + TypeScript]
        A2[Material-UI 组件]
        A3[WebSocket 客户端]
        A4[Vite 构建工具]
    end
    
    subgraph "后端服务层 Service"
        B1[FastAPI 网关]
        B2[认证服务]
        B3[学习训练服务]
        B4[手语识别服务]
        B5[LLM对话服务]
    end
    
    subgraph "AI推理层 AI Engine"
        C1[Mind-VAC CSLR]
        C2[I3D孤立识别]
        C3[通义千问 LLM]
        C4[MediaPipe 可选]
    end
    
    subgraph "数据存储层 Storage"
        D1[(SQLite/PostgreSQL)]
        D2[Redis 缓存]
        D3[文件存储 uploads/]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 -.构建.-> A1
    
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5
    
    B4 --> C1
    B4 --> C2
    B5 --> C3
    B4 -.可选.-> C4
    
    B2 --> D1
    B3 --> D1
    B4 --> D2
    B4 --> D3
    
    style A1 fill:#61DAFB,stroke:#333,color:#000
    style B1 fill:#009688,stroke:#333,color:#fff
    style C1 fill:#FF6B35,stroke:#333,color:#fff
    style D1 fill:#4CAF50,stroke:#333,color:#fff
```

## 核心能力流程图

```mermaid
flowchart TD
    Start([用户访问 SignAvatar]) --> Choice{选择功能}
    
    Choice -->|连续手语识别| CSLR[上传手语视频]
    Choice -->|孤立手语识别| ISL[录制短手语]
    Choice -->|学习训练| Learn[选择课程]
    
    CSLR --> CSLR1[视频解码]
    CSLR1 --> CSLR2[Mind-VAC推理]
    CSLR2 --> CSLR3[CTC解码 Gloss序列]
    CSLR3 --> CSLR4{启用LLM?}
    CSLR4 -->|是| CSLR5[通义千问翻译]
    CSLR4 -->|否| CSLR6[词典映射]
    CSLR5 --> CSLR7[生成中英文句子]
    CSLR6 --> CSLR7
    CSLR7 --> CSLR8[输出SRT字幕]
    
    ISL --> ISL1[关键点提取]
    ISL1 --> ISL2[I3D模型推理]
    ISL2 --> ISL3[Top-K分类结果]
    ISL3 --> ISL4[练习建议反馈]
    
    Learn --> Learn1[课程列表展示]
    Learn1 --> Learn2[视频/互动学习]
    Learn2 --> Learn3[完成度追踪]
    Learn3 --> Learn4[成就解锁]
    Learn4 --> Learn5[个性化推荐]
    
    CSLR8 --> End([任务完成])
    ISL4 --> End
    Learn5 --> End
    
    style Start fill:#E1F5FE,stroke:#01579B
    style End fill:#C8E6C9,stroke:#1B5E20
    style CSLR fill:#FFE0B2,stroke:#E65100
    style ISL fill:#FFF9C4,stroke:#F57F17
    style Learn fill:#F3E5F5,stroke:#4A148C
```

## 服务依赖关系图

```mermaid
graph TD
    SM[ServiceManager<br/>服务管理器] --> FM[FileManager<br/>文件管理]
    SM --> MP[MediaPipeService<br/>关键点提取]
    SM --> CSLR[CSLRService<br/>连续识别]
    SM --> SR[SignRecognitionService<br/>识别任务管理]
    SM --> LT[LearningTrainingService<br/>学习训练]
    SM --> ISL[IsolatedSignService<br/>孤立识别]
    
    SR --> CSLR
    SR --> MP
    CSLR --> MV[MindVacEngine<br/>Mind-VAC引擎]
    MV --> QW[QwenAPI<br/>通义千问]
    
    LT --> ACH[AchievementService<br/>成就系统]
    LT --> CM[CourseManagement<br/>课程管理]
    
    ISL --> MS[MindSpore<br/>I3D模型]
    
    SR --> DB[(Database)]
    LT --> DB
    SR --> REDIS[(Redis Cache)]
    SR --> FS[File Storage<br/>文件存储]
    
    style SM fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff
    style SR fill:#4ECDC4,stroke:#0E9594,stroke-width:2px,color:#fff
    style CSLR fill:#95E1D3,stroke:#38A69D,stroke-width:2px
    style MV fill:#F38181,stroke:#E74C3C,stroke-width:2px,color:#fff
    style LT fill:#AA96DA,stroke:#6C5B7B,stroke-width:2px,color:#fff
    style DB fill:#FCBAD3,stroke:#C06C84,stroke-width:2px
```

## 用户交互流程图

```mermaid
sequenceDiagram
    actor User as 👤 用户
    participant FE as 🌐 前端界面
    participant API as 🚀 FastAPI网关
    participant CSLR as 🧠 CSLR服务
    participant MV as 🔮 Mind-VAC
    participant LLM as 🤖 通义千问
    participant DB as 💾 数据库
    
    User->>FE: 上传手语视频
    FE->>API: POST /api/sign-recognition/upload-video
    API->>CSLR: 创建识别任务
    CSLR->>DB: 保存任务状态
    CSLR-->>API: 返回任务ID
    API-->>FE: task_id
    FE-->>User: 显示上传成功
    
    CSLR->>CSLR: 视频解码(OpenCV)
    CSLR->>MV: 传入RGB帧序列
    MV->>MV: 预处理+推理
    MV->>MV: CTC解码
    MV-->>CSLR: Gloss序列
    
    alt LLM已启用
        CSLR->>LLM: 翻译Gloss
        LLM-->>CSLR: 中英文句子
    end
    
    CSLR->>CSLR: 生成SRT字幕
    CSLR->>DB: 更新结果
    
    User->>FE: 轮询任务状态
    FE->>API: GET /api/sign-recognition/status/{task_id}
    API->>DB: 查询任务
    DB-->>API: 任务结果
    API-->>FE: JSON结果
    FE-->>User: 展示翻译文本+字幕
```

## 数据流架构图

```mermaid
flowchart LR
    subgraph "输入数据"
        V[📹 手语视频]
        K[✋ 关键点流]
        U[👤 用户操作]
    end
    
    subgraph "处理层"
        P1[视频解码]
        P2[帧预处理]
        P3[特征提取]
        P4[序列建模]
        P5[CTC解码]
    end
    
    subgraph "AI推理"
        AI1[MindSpore<br/>CSLR模型]
        AI2[I3D<br/>孤立识别]
        AI3[LLM<br/>后处理]
    end
    
    subgraph "输出结果"
        O1[📄 Gloss序列]
        O2[💬 自然语言]
        O3[📝 SRT字幕]
        O4[📊 学习报告]
    end
    
    V --> P1
    P1 --> P2
    K --> P3
    P2 --> P3
    P3 --> P4
    P4 --> AI1
    P4 --> AI2
    AI1 --> P5
    P5 --> O1
    O1 --> AI3
    AI3 --> O2
    O2 --> O3
    U --> O4
    
    style V fill:#FFD6E8,stroke:#FF69B4
    style AI1 fill:#FFDAB9,stroke:#FF8C00
    style O2 fill:#B0E0E6,stroke:#4682B4
```

## 技术选型决策树

```mermaid
graph TD
    A{项目需求} --> B{识别任务类型}
    B -->|连续手语| C[Mind-VAC + MindSpore]
    B -->|孤立手语| D[I3D + MindSpore]
    B -->|实时流式| E[WebSocket + MediaPipe]
    
    C --> F{部署环境}
    F -->|云端| G[GPU + 容器化]
    F -->|边缘设备| H[CPU优化 + 量化]
    
    D --> I{准确率要求}
    I -->|高| J[Top-10集成]
    I -->|一般| K[Top-5快速]
    
    E --> L{延迟要求}
    L -->|<500ms| M[关键点缓存]
    L -->|<2s| N[批量推理]
    
    C --> O{翻译质量}
    O -->|高| P[通义千问 qwen-plus]
    O -->|快速| Q[本地词典映射]
    
    style A fill:#FFE5E5,stroke:#FF6B6B,stroke-width:3px
    style C fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style P fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

## 部署架构图

```mermaid
graph TB
    subgraph "客户端 Client"
        Web[🌐 Web浏览器<br/>React前端]
        Mobile[📱 移动端<br/>规划中]
    end
    
    subgraph "负载均衡 Load Balancer"
        LB[Nginx / Caddy]
    end
    
    subgraph "应用服务器 Application"
        API1[FastAPI 实例1]
        API2[FastAPI 实例2]
        API3[FastAPI 实例N]
    end
    
    subgraph "AI推理服务 AI Inference"
        GPU1[GPU推理节点]
        CPU1[CPU推理节点]
        CPU2[CPU推理节点]
    end
    
    subgraph "数据层 Data"
        PG[(PostgreSQL<br/>主数据库)]
        RD[(Redis<br/>缓存+会话)]
        S3[对象存储<br/>MinIO/S3]
    end
    
    subgraph "监控运维 Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        LOG[日志聚合]
    end
    
    Web --> LB
    Mobile -.未来.-> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> GPU1
    API2 --> CPU1
    API3 --> CPU2
    
    API1 --> PG
    API2 --> PG
    API3 --> PG
    API1 --> RD
    API2 --> RD
    API3 --> RD
    API1 --> S3
    API2 --> S3
    API3 --> S3
    
    API1 -.metrics.-> PROM
    API2 -.metrics.-> PROM
    API3 -.metrics.-> PROM
    PROM --> GRAF
    API1 -.logs.-> LOG
    API2 -.logs.-> LOG
    API3 -.logs.-> LOG
    
    style Web fill:#61DAFB,stroke:#333
    style LB fill:#269AC7,stroke:#333,color:#fff
    style API1 fill:#009688,stroke:#333,color:#fff
    style GPU1 fill:#FF6B35,stroke:#333,color:#fff
    style PG fill:#336791,stroke:#333,color:#fff
    style PROM fill:#E6522C,stroke:#333,color:#fff
```

---

**说明**: 
- 这些图表展示了 SignAvatar 系统从不同视角的架构设计
- 可以根据具体PPT页面需求选择合适的图表使用
- 所有图表均可在支持 Mermaid 的环境中渲染(如 GitHub、VS Code、Typora等)
