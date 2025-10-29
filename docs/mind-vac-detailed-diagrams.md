# Mind-VAC CSLR 引擎与 LLM 增强详细图示

## 1. Mind-VAC CSLR 完整推理流程图

```mermaid
flowchart TB
    subgraph "输入阶段 Input Stage"
        A1[📹 用户上传视频<br/>Video Upload] --> A2[OpenCV 视频解码<br/>Video Decoding]
        A2 --> A3[帧采样 25fps<br/>Frame Sampling]
        A3 --> A4{帧数检查<br/>Frame Count}
        A4 -->|< 64帧| A5[尾部填充<br/>Tail Padding]
        A4 -->|≥ 64帧| A6[正常处理<br/>Normal Process]
        A5 --> A6
    end
    
    subgraph "预处理阶段 Preprocessing"
        A6 --> B1[CenterCrop 224x224<br/>中心裁剪]
        B1 --> B2[RGB归一化<br/>Normalize to range -1 to 1]
        B2 --> B3[时序对齐<br/>Temporal Alignment]
        B3 --> B4[前后填充<br/>Leading+Trailing Pad]
        B4 --> B5[对齐到4的倍数<br/>Align to Multiple of 4]
        B5 --> B6[转为Tensor<br/>Convert to MindSpore Tensor]
    end
    
    subgraph "Mind-VAC 模型推理 Model Inference"
        B6 --> C1[ResNet18 特征提取<br/>Feature Extraction<br/>输出: T×512]
        C1 --> C2[1D卷积层<br/>Conv1D Temporal<br/>输出: T'×1024]
        C2 --> C3[双向LSTM<br/>Bidirectional LSTM<br/>hidden_size=1024]
        C3 --> C4[全连接分类头<br/>FC Classifier<br/>输出: T'×num_classes]
        C4 --> C5[Softmax归一化<br/>Softmax]
    end
    
    subgraph "CTC解码阶段 CTC Decoding"
        C5 --> D1[Beam Search<br/>束搜索<br/>beam_width=10]
        D1 --> D2[去除Blank标记<br/>Remove CTC Blank]
        D2 --> D3[去除重复<br/>Collapse Repeats]
        D3 --> D4[索引映射词表<br/>Index to Gloss Mapping]
        D4 --> D5[📄 Gloss序列<br/>例: __ON__ LIEB ZUSCHAUER ABEND]
    end
    
    subgraph "后处理阶段 Post-processing"
        D5 --> E1{启用LLM?<br/>LLM Enabled?}
        E1 -->|否 No| E2[词典映射<br/>Dictionary Mapping]
        E1 -->|是 Yes| E3[调用通义千问<br/>Call Qwen API]
        E2 --> E4[基线翻译<br/>Baseline Translation]
        E3 --> E5[LLM增强翻译<br/>Enhanced Translation]
        E4 --> E6[🎬 生成SRT字幕<br/>Generate Subtitles]
        E5 --> E6
        E6 --> E7[💾 保存结果<br/>JSON + TXT + SRT]
    end
    
    E7 --> F1[✅ 返回给用户<br/>Return to User]
    
    style A1 fill:#FFE5E5,stroke:#FF6B6B,stroke-width:3px
    style C3 fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style D1 fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    style E3 fill:#E8F5E9,stroke:#4CAF50,stroke-width:3px
    style F1 fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
```

## 2. Mind-VAC 模型架构详细图

```mermaid
graph TB
    subgraph "输入层 Input Layer"
        I1[视频帧序列<br/>Shape: B × T × 3 × 224 × 224<br/>B=batch, T=时序长度]
    end
    
    subgraph "骨干网络 Backbone: ResNet18"
        R1[Conv2D 7×7, stride=2<br/>输出: 64 channels]
        R2[MaxPool 3×3, stride=2]
        R3[ResBlock Layer1<br/>64 channels × 2]
        R4[ResBlock Layer2<br/>128 channels × 2]
        R5[ResBlock Layer3<br/>256 channels × 2]
        R6[ResBlock Layer4<br/>512 channels × 2]
        R7[AdaptiveAvgPool<br/>输出: B × T × 512]
    end
    
    subgraph "时序建模 Temporal Modeling"
        T1[Conv1D 时序卷积<br/>kernel_size=5<br/>输入: 512 → 输出: 1024]
        T2[BatchNorm1D]
        T3[ReLU激活]
        T4[Dropout p=0.5]
        T5[LSTM 双向<br/>input_size=1024<br/>hidden_size=1024<br/>num_layers=2]
        T6[输出: B × T' × 2048<br/>前向+后向拼接]
    end
    
    subgraph "分类头 Classification Head"
        C1[Linear 全连接<br/>2048 → num_classes]
        C2[输出: B × T' × num_classes<br/>num_classes=1296<br/>1295个gloss + 1个blank]
    end
    
    I1 --> R1
    R1 --> R2
    R2 --> R3
    R3 --> R4
    R4 --> R5
    R5 --> R6
    R6 --> R7
    
    R7 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    T5 --> T6
    
    T6 --> C1
    C1 --> C2
    
    style I1 fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style R7 fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
    style T5 fill:#F3E5F5,stroke:#6A1B9A,stroke-width:3px
    style C2 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
```

## 3. CTC 解码算法详细流程

```mermaid
flowchart LR
    subgraph "CTC Logits 输入"
        A[模型输出<br/>Shape: T × num_classes<br/>例: 64 × 1296]
    end
    
    subgraph "Beam Search 解码"
        B1[初始化Beam<br/>候选路径队列]
        B2[遍历每个时间步 t]
        B3[对每条路径扩展<br/>Top-k候选]
        B4[计算路径概率<br/>log_prob累加]
        B5[保留Top-B路径<br/>beam_width=10]
        B6{是否到末尾?}
        B6 -->|否| B2
        B6 -->|是| B7[选择最优路径]
    end
    
    subgraph "后处理 Post-process"
        C1[去除CTC Blank<br/>ID=0的标记]
        C2[合并连续重复<br/>例: AAA → A]
        C3[索引转Gloss<br/>ID → 词表映射]
        C4[过滤特殊符号<br/>__ON__, __OFF__等]
    end
    
    subgraph "输出 Output"
        D[Gloss序列<br/>+ 位置信息<br/>+ 置信度]
    end
    
    A --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    B7 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D
    
    style A fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style B5 fill:#E1F5FE,stroke:#0277BD,stroke-width:3px
    style C2 fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style D fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px
```

## 4. LLM 增强翻译完整流程

```mermaid
sequenceDiagram
    participant CSLR as Mind-VAC引擎
    participant Cache as Redis缓存
    participant LLM as 通义千问API
    participant Post as 后处理模块
    participant User as 用户
    
    CSLR->>CSLR: CTC解码得到Gloss序列
    Note over CSLR: 例: __ON__ LIEB ZUSCHAUER<br/>ABEND WINTER NULL loc-REGION<br/>UEBERSCHWEMMUNG AMERIKA
    
    CSLR->>Cache: 查询缓存<br/>Key: hash(gloss_sequence)
    alt 缓存命中
        Cache-->>CSLR: 返回缓存翻译
        Note over Cache: TTL=24小时<br/>节省API调用
    else 缓存未命中
        CSLR->>LLM: 构造Prompt并调用
        Note over LLM: 模型: qwen-plus<br/>温度: 0.3<br/>max_tokens: 200
        
        rect rgb(255, 243, 224)
            Note right of LLM: Prompt内容:<br/>你是手语翻译专家...<br/>输入Gloss: xxx<br/>输出JSON格式...
        end
        
        LLM->>LLM: GPT推理生成
        LLM-->>CSLR: 返回JSON结果
        
        rect rgb(232, 245, 233)
            Note right of CSLR: 返回格式:<br/>{<br/>  "chinese": "...",<br/>  "english": "...",<br/>  "confidence": "high",<br/>  "explanation": "..."<br/>}
        end
        
        CSLR->>Cache: 存入缓存
    end
    
    CSLR->>Post: 传递翻译结果
    Post->>Post: 标点符号修正
    Post->>Post: 分句处理
    Post->>Post: 生成SRT时间轴
    Post->>User: 返回最终结果
    
    Note over User: 显示:<br/>原始Gloss<br/>中文翻译<br/>英文翻译<br/>SRT字幕
```

## 5. LLM Prompt 工程详解

```mermaid
graph TB
    subgraph "Prompt 构造流程"
        P1[系统角色设定<br/>System Role]
        P2[任务描述<br/>Task Description]
        P3[输入格式说明<br/>Input Format]
        P4[输出格式约束<br/>Output Format]
        P5[Few-shot示例<br/>Examples Optional]
    end
    
    subgraph "实际Prompt示例"
        E1["系统: 你是专业的手语翻译专家,<br/>精通德语手语(DGS)到中英文的转换。"]
        E2["任务: 将以下手语Gloss序列翻译为<br/>流畅的中文和英文句子。"]
        E3["输入Gloss: __ON__ LIEB ZUSCHAUER<br/>ABEND WINTER NULL loc-REGION<br/>UEBERSCHWEMMUNG AMERIKA"]
        E4["要求输出JSON格式:<br/>{<br/>  'chinese': '中文句子',<br/>  'english': 'English sentence',<br/>  'confidence': 'high/medium/low',<br/>  'explanation': '简要说明'<br/>}"]
        E5["注意事项:<br/>1. 保持语义完整<br/>2. 去除无意义标记<br/>3. 符合语法规范<br/>4. 置信度评估依据充分"]
    end
    
    subgraph "LLM输出示例"
        O1["{<br/>  'chinese': '亲爱的观众,晚上好。<br/>冬季该地区在美国发生了洪水。',<br/>  'english': 'Dear viewers, good evening.<br/>Floods occurred in the region<br/>in America during winter.',<br/>  'confidence': 'high',<br/>  'explanation': '识别到问候语和<br/>新闻播报内容,语义清晰。'<br/>}"]
    end
    
    P1 --> E1
    P2 --> E2
    P3 --> E3
    P4 --> E4
    P5 --> E5
    
    E1 --> O1
    E2 --> O1
    E3 --> O1
    E4 --> O1
    E5 --> O1
    
    style P1 fill:#FFEBEE,stroke:#C62828
    style P4 fill:#E3F2FD,stroke:#1565C0
    style O1 fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px
```

## 6. 通义千问 API 调用架构

```mermaid
graph TB
    subgraph "调用端 Client Side"
        A1[Mind-VAC引擎] --> A2[QwenAPI封装类]
        A2 --> A3[构造HTTP请求]
        A3 --> A4[添加认证Header<br/>API-Key]
    end
    
    subgraph "网络层 Network"
        A4 --> B1[HTTPS加密传输]
        B1 --> B2[阿里云DashScope<br/>API网关]
    end
    
    subgraph "服务端 Server Side"
        B2 --> C1[请求验证]
        C1 --> C2[模型路由]
        C2 --> C3{选择模型}
        C3 -->|快速| C4[qwen-turbo<br/>速度快,成本低]
        C3 -->|平衡| C5[qwen-plus<br/>推荐,性价比高]
        C3 -->|高精度| C6[qwen-max<br/>最强性能]
        C4 --> C7[推理生成]
        C5 --> C7
        C6 --> C7
        C7 --> C8[返回JSON响应]
    end
    
    subgraph "响应处理 Response Handling"
        C8 --> D1[解析JSON]
        D1 --> D2{成功?}
        D2 -->|是| D3[提取翻译结果]
        D2 -->|否| D4[错误处理]
        D4 --> D5{重试次数<3?}
        D5 -->|是| B1
        D5 -->|否| D6[降级到词典映射]
        D3 --> D7[返回给Mind-VAC]
        D6 --> D7
    end
    
    style A2 fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style C5 fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px
    style D4 fill:#FFEBEE,stroke:#C62828,stroke-width:2px
```

## 7. Gloss 到自然语言的转换对比

```mermaid
graph LR
    subgraph "原始Gloss"
        G["__ON__ LIEB ZUSCHAUER<br/>ABEND WINTER NULL<br/>loc-REGION UEBERSCHWEMMUNG<br/>AMERIKA"]
    end
    
    subgraph "词典映射 Dictionary"
        D1[简单查表]
        D2[逐词翻译]
        D3[拼接结果]
        D4["输出: 开启 亲爱 观众<br/>晚上 冬天 空 地区<br/>洪水 美国"]
        D5["问题:<br/>❌ 不流畅<br/>❌ 无语法<br/>❌ 保留噪声"]
    end
    
    subgraph "LLM增强 Enhanced"
        L1[语义理解]
        L2[上下文推理]
        L3[语法重构]
        L4["输出中文:<br/>亲爱的观众,晚上好。<br/>冬季该地区在美国<br/>发生了洪水。"]
        L5["输出英文:<br/>Dear viewers, good evening.<br/>Floods occurred in the region<br/>in America during winter."]
        L6["优势:<br/>✅ 流畅自然<br/>✅ 符合语法<br/>✅ 过滤噪声<br/>✅ 双语输出"]
    end
    
    G --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5
    
    G --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    
    style G fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style D5 fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style L6 fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px
```

## 8. SRT 字幕生成流程

```mermaid
flowchart TB
    subgraph "输入数据 Input"
        A1[Gloss序列 + 位置信息]
        A2[视频FPS信息]
        A3[LLM翻译结果]
    end
    
    subgraph "时间戳计算 Timestamp"
        B1[起始帧号 start_frame]
        B2[结束帧号 end_frame]
        B3[计算秒数<br/>time = frame / fps]
        B4[格式化时间<br/>HH:MM:SS,mmm]
    end
    
    subgraph "字幕分段 Segmentation"
        C1[按Gloss分组]
        C2[或按时间窗口<br/>例: 5秒一段]
        C3[或按句子结构<br/>标点符号分割]
    end
    
    subgraph "SRT格式生成 Format"
        D1[序号<br/>1, 2, 3...]
        D2[时间轴<br/>00:00:01,000 --> 00:00:05,500]
        D3[字幕文本<br/>亲爱的观众,晚上好。]
        D4[空行分隔]
    end
    
    subgraph "输出文件 Output"
        E1[task_id.srt<br/>UTF-8编码]
        E2[静态文件服务<br/>/sign_results/task_id.srt]
    end
    
    A1 --> B1
    A2 --> B3
    A3 --> C1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C1
    C1 --> C2
    C2 --> C3
    
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    D4 --> E1
    E1 --> E2
    
    style A3 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style D3 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style E1 fill:#FFF3E0,stroke:#F57C00,stroke-width:3px
```

## 9. 实时识别 WebSocket 模式

```mermaid
sequenceDiagram
    participant Client as 🌐 前端客户端
    participant WS as 🔌 WebSocket服务
    participant MP as 📹 MediaPipe
    participant CSLR as 🧠 CSLR服务
    participant Buffer as 📦 序列缓冲区
    
    Client->>WS: 建立WebSocket连接
    WS-->>Client: 连接确认 + 配置信息
    
    loop 实时帧流
        Client->>Client: 摄像头捕获帧
        Client->>MP: 提取关键点
        MP-->>Client: 543个关键点坐标
        Client->>WS: 发送关键点JSON<br/>{type:"landmarks",payload:{...}}
        WS->>Buffer: 追加到序列缓冲
        
        alt 缓冲区达到窗口长度
            Buffer->>CSLR: 传入序列(64帧)
            CSLR->>CSLR: 推理+CTC解码
            CSLR-->>WS: 返回Gloss预测
            WS-->>Client: {type:"recognition_result",<br/>payload:{text,confidence,gloss}}
            Client->>Client: 更新UI显示
        end
    end
    
    Client->>WS: 关闭连接
    WS->>Buffer: 清空缓冲区
    WS-->>Client: 连接关闭确认
```

## 10. 性能优化关键点

```mermaid
graph TB
    subgraph "模型层优化 Model Optimization"
        M1[图模式编译<br/>GRAPH_MODE]
        M2[算子融合<br/>Operator Fusion]
        M3[内存复用<br/>Memory Pool]
    end
    
    subgraph "推理加速 Inference Speed"
        I1[批量推理<br/>Batch Processing]
        I2[动态shape<br/>Dynamic Shape]
        I3[INT8量化<br/>Quantization]
        I4[模型蒸馏<br/>Distillation]
    end
    
    subgraph "缓存策略 Caching"
        C1[Redis缓存<br/>LLM翻译结果]
        C2[本地缓存<br/>模型权重预加载]
        C3[结果缓存<br/>相似视频识别]
    end
    
    subgraph "并发优化 Concurrency"
        P1[异步处理<br/>AsyncIO]
        P2[任务队列<br/>Celery/RQ]
        P3[多进程推理<br/>Worker Pool]
    end
    
    subgraph "硬件加速 Hardware"
        H1[GPU推理<br/>CUDA]
        H2[NPU推理<br/>Ascend]
        H3[SIMD指令<br/>AVX2/AVX512]
    end
    
    M1 --> I1
    M2 --> I2
    M3 --> I3
    I3 --> I4
    
    I1 --> C1
    I2 --> C2
    I4 --> C3
    
    C1 --> P1
    C2 --> P2
    C3 --> P3
    
    P1 --> H1
    P2 --> H2
    P3 --> H3
    
    style M1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style I3 fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style C1 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style H2 fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
```

## 11. 错误处理与降级策略

```mermaid
flowchart TD
    A[视频输入] --> B{帧数检查}
    B -->|< 8帧| C[❌ 拒绝: 视频过短]
    B -->|正常| D[开始处理]
    
    D --> E{MindSpore可用?}
    E -->|否| F[❌ 返回503错误]
    E -->|是| G[Mind-VAC推理]
    
    G --> H{推理成功?}
    H -->|否| I{重试次数<3?}
    I -->|是| G
    I -->|否| J[降级: 使用备用模型]
    H -->|是| K[CTC解码]
    
    J --> K
    K --> L{LLM启用?}
    L -->|是| M{API可用?}
    M -->|否| N{网络重试<3?}
    N -->|是| M
    N -->|否| O[降级: 词典映射]
    M -->|是| P[LLM翻译]
    L -->|否| O
    
    P --> Q{翻译质量检查}
    Q -->|低质量| O
    Q -->|正常| R[返回结果]
    O --> R
    
    R --> S[生成SRT]
    S --> T[✅ 完成]
    
    style C fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    style F fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    style O fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style P fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style T fill:#B2DFDB,stroke:#00695C,stroke-width:3px
```

## 12. Mind-VAC vs 传统方法对比

```mermaid
graph TB
    subgraph "传统CNN-RNN方法"
        T1[2D CNN<br/>逐帧特征提取]
        T2[手工特征<br/>HOG/SIFT]
        T3[简单RNN/GRU<br/>时序建模]
        T4[CTC或Seq2Seq<br/>解码]
        T5[问题:<br/>❌ 时空特征分离<br/>❌ 梯度消失<br/>❌ 对齐困难]
    end
    
    subgraph "Mind-VAC创新"
        M1[ResNet18<br/>深度特征]
        M2[1D Conv + BiLSTM<br/>增强时序建模]
        M3[Visual Alignment<br/>视觉对齐约束]
        M4[CTC + Beam Search<br/>高效解码]
        M5[优势:<br/>✅ 端到端训练<br/>✅ 时空联合建模<br/>✅ 鲁棒性强<br/>✅ 可迁移性好]
    end
    
    subgraph "本项目增强"
        P1[MindSpore适配<br/>跨平台部署]
        P2[LLM后处理<br/>翻译增强]
        P3[实时+离线<br/>双模式]
        P4[CPU优化<br/>边缘友好]
        P5[亮点:<br/>🔥 生产级工程<br/>🔥 即插即用<br/>🔥 开源开放]
    end
    
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M5
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    
    T5 -.演进.-> M1
    M5 -.升级.-> P1
    
    style T5 fill:#FFCDD2,stroke:#C62828
    style M5 fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style P5 fill:#B2DFDB,stroke:#00695C,stroke-width:3px
```

---

## 使用建议

### PPT第5页 (Mind-VAC CSLR引擎):
- **主图**: 图1 完整推理流程图
- **辅助**: 图2 模型架构详细图
- **备选**: 图10 性能优化关键点

### PPT第6页 (LLM增强翻译):
- **主图**: 图4 LLM增强翻译完整流程
- **对比**: 图7 Gloss到自然语言转换对比
- **技术细节**: 图5 Prompt工程详解

### PPT第8页 (技术创新):
- **对比图**: 图12 Mind-VAC vs 传统方法
- **优化图**: 图10 性能优化关键点
- **容错图**: 图11 错误处理与降级策略

这些图表完整展示了 Mind-VAC + LLM 的技术细节和工程实践! 🎨
