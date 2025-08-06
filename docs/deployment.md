# SignAvatar Web 部署指南

## 部署概述

SignAvatar Web 支持多种部署方式，从开发环境到生产环境的完整部署方案。

## 快速部署

### 使用 Docker Compose (推荐)

1. **准备环境**
```bash
# 确保已安装 Docker 和 Docker Compose
docker --version
docker-compose --version
```

2. **克隆项目**
```bash
git clone <repository-url>
cd signavatar-web
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，设置生产环境配置
```

4. **一键部署**
```bash
chmod +x deploy.sh
./deploy.sh
```

5. **验证部署**
- 前端: http://localhost:3000
- 后端 API: http://localhost:8000
- 监控面板: http://localhost:3001 (admin/admin)

## 详细部署步骤

### 1. 环境准备

#### 系统要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+
- **内存**: 最少 4GB，推荐 8GB+
- **存储**: 最少 20GB 可用空间
- **网络**: 稳定的互联网连接

#### 依赖安装
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose git curl

# CentOS/RHEL
sudo yum install -y docker docker-compose git curl
sudo systemctl start docker
sudo systemctl enable docker

# 添加用户到 docker 组
sudo usermod -aG docker $USER
```

### 2. 项目配置

#### 环境变量配置
```bash
# .env 文件示例
APP_NAME=SignAvatar Web
VERSION=1.0.0
DEBUG=false

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置 (如果使用)
DATABASE_URL=postgresql://user:password@localhost:5432/signavatar

# 安全配置
SECRET_KEY=your-super-secret-key-here
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# AI 模型配置
CSLR_MODEL_PATH=models/cslr_model.mindir
CSLR_CONFIDENCE_THRESHOLD=0.6

# 监控配置
ENABLE_METRICS=true
METRICS_PORT=9090
```

#### SSL 证书配置
```bash
# 创建 SSL 目录
mkdir -p ssl

# 使用 Let's Encrypt (推荐)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# 复制证书
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/
```

### 3. 服务部署

#### Docker Compose 配置
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    build: ./backend
    environment:
      - DEBUG=false
      - LOG_LEVEL=WARNING
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

#### 启动服务
```bash
# 生产环境部署
docker-compose -f docker-compose.prod.yml up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

### 4. 反向代理配置

#### Nginx 配置
```nginx
# /etc/nginx/sites-available/signavatar
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;

    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # 前端静态文件
    location / {
        proxy_pass http://frontend:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API 代理
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket 代理
    location /ws/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 5. 监控配置

#### Prometheus 配置
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'signavatar-backend'
    static_configs:
      - targets: ['backend:9090']

  - job_name: 'signavatar-frontend'
    static_configs:
      - targets: ['frontend:80']
```

#### Grafana 仪表板
```json
{
  "dashboard": {
    "title": "SignAvatar Web Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

## 生产环境优化

### 1. 性能优化

#### 后端优化
```python
# 生产环境配置
WORKERS = 4  # CPU 核心数
MAX_CONNECTIONS = 1000
KEEPALIVE_TIMEOUT = 65
```

#### 前端优化
```javascript
// 生产构建优化
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          three: ['three', '@react-three/fiber'],
          ui: ['@mui/material']
        }
      }
    }
  }
})
```

### 2. 安全配置

#### 防火墙设置
```bash
# UFW 配置
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

#### 安全头配置
```nginx
# 安全头
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### 3. 备份策略

#### 数据备份
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/signavatar"

# 备份数据库
docker exec postgres pg_dump -U user signavatar > $BACKUP_DIR/db_$DATE.sql

# 备份模型文件
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# 备份配置文件
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env docker-compose.yml
```

#### 自动备份
```bash
# 添加到 crontab
0 2 * * * /path/to/backup.sh
```

## 故障排除

### 常见问题

1. **容器启动失败**
```bash
# 查看详细日志
docker-compose logs backend
docker-compose logs frontend

# 检查容器状态
docker ps -a
```

2. **端口冲突**
```bash
# 查看端口占用
netstat -tulpn | grep :8000
lsof -i :8000

# 修改端口配置
vim docker-compose.yml
```

3. **SSL 证书问题**
```bash
# 检查证书有效性
openssl x509 -in ssl/fullchain.pem -text -noout

# 更新证书
sudo certbot renew
```

4. **性能问题**
```bash
# 监控资源使用
docker stats
htop

# 调整资源限制
docker-compose up -d --scale backend=2
```

### 日志管理

#### 日志配置
```yaml
# docker-compose.yml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

#### 日志查看
```bash
# 实时日志
docker-compose logs -f --tail=100 backend

# 错误日志
docker-compose logs backend | grep ERROR

# 日志轮转
logrotate /etc/logrotate.d/docker-containers
```

## 扩展部署

### 1. 负载均衡

#### HAProxy 配置
```
backend signavatar_backend
    balance roundrobin
    server backend1 backend1:8000 check
    server backend2 backend2:8000 check
```

### 2. 集群部署

#### Docker Swarm
```bash
# 初始化 Swarm
docker swarm init

# 部署服务栈
docker stack deploy -c docker-compose.yml signavatar
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: signavatar-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: signavatar-backend
  template:
    metadata:
      labels:
        app: signavatar-backend
    spec:
      containers:
      - name: backend
        image: signavatar/backend:latest
        ports:
        - containerPort: 8000
```

## 维护指南

### 1. 更新部署

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build

# 滚动更新
docker-compose up -d
```

### 2. 健康检查

```bash
# 自动健康检查脚本
#!/bin/bash
curl -f http://localhost:8000/api/health || exit 1
```

### 3. 监控告警

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'
```

通过以上配置，您可以成功部署 SignAvatar Web 系统到生产环境，并确保系统的稳定性和可扩展性。
