# 手语学习训练系统部署指南

## 系统要求

### 硬件要求
- **CPU**: 4核心以上，推荐8核心
- **内存**: 8GB以上，推荐16GB
- **存储**: 50GB以上可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+ / macOS 10.15+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.25+

## 快速部署

### 1. 克隆项目
```bash
git clone https://github.com/your-org/sign-language-learning.git
cd sign-language-learning
```

### 2. 环境配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

### 3. 启动服务
```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 访问系统
- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **系统监控**: http://localhost:9090 (Prometheus)
- **可视化面板**: http://localhost:3001 (Grafana)

## 详细配置

### 环境变量说明

```bash
# 应用配置
APP_NAME=手语学习训练系统
APP_VERSION=1.0.0
APP_ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-change-this

# 数据库配置
DATABASE_URL=sqlite:///./data/sign_language.db

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# 文件上传配置
MAX_FILE_SIZE=100MB
UPLOAD_DIR=./uploads

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# 监控配置
GRAFANA_PASSWORD=admin123
PROMETHEUS_RETENTION=30d

# SSL配置 (生产环境)
SSL_CERT_PATH=./nginx/ssl/cert.pem
SSL_KEY_PATH=./nginx/ssl/key.pem
```

### 数据库初始化

```bash
# 进入后端容器
docker-compose exec backend bash

# 运行数据库迁移
python -m backend.scripts.migrate

# 创建管理员用户
python -m backend.scripts.create_admin --username admin --email admin@example.com
```

### SSL证书配置

#### 使用Let's Encrypt (推荐)
```bash
# 安装certbot
sudo apt-get install certbot

# 获取证书
sudo certbot certonly --standalone -d your-domain.com

# 复制证书到项目目录
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./nginx/ssl/key.pem
```

#### 使用自签名证书 (开发环境)
```bash
# 创建SSL目录
mkdir -p nginx/ssl

# 生成自签名证书
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=CN/ST=Beijing/L=Beijing/O=SignLanguage/CN=localhost"
```

## 生产环境部署

### 1. 服务器准备
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 创建应用用户
sudo useradd -m -s /bin/bash signlang
sudo usermod -aG docker signlang
```

### 2. 防火墙配置
```bash
# 开放必要端口
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### 3. 系统服务配置
```bash
# 创建systemd服务文件
sudo tee /etc/systemd/system/sign-language.service > /dev/null <<EOF
[Unit]
Description=Sign Language Learning System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/signlang/sign-language-learning
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=signlang

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl enable sign-language.service
sudo systemctl start sign-language.service
```

### 4. 备份策略
```bash
# 创建备份脚本
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/sign-language"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
docker-compose exec -T backend python -m backend.scripts.backup_db > $BACKUP_DIR/db_$DATE.sql

# 备份上传文件
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz uploads/

# 清理旧备份 (保留30天)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x backup.sh

# 添加到crontab (每天凌晨2点备份)
echo "0 2 * * * /home/signlang/sign-language-learning/backup.sh" | crontab -
```

## 监控和维护

### 健康检查
```bash
# 检查所有服务状态
docker-compose ps

# 检查系统健康
curl http://localhost:8000/api/system/health

# 查看资源使用情况
docker stats
```

### 日志管理
```bash
# 查看应用日志
docker-compose logs backend
docker-compose logs frontend

# 实时跟踪日志
docker-compose logs -f --tail=100

# 清理日志
docker system prune -f
```

### 性能优化
```bash
# 优化Docker镜像
docker image prune -f

# 清理未使用的容器
docker container prune -f

# 清理未使用的网络
docker network prune -f
```

## 故障排除

### 常见问题

#### 1. 容器启动失败
```bash
# 检查容器日志
docker-compose logs [service_name]

# 检查端口占用
netstat -tulpn | grep :8000

# 重新构建镜像
docker-compose build --no-cache [service_name]
```

#### 2. 数据库连接失败
```bash
# 检查数据库文件权限
ls -la data/

# 重置数据库
rm data/sign_language.db
docker-compose restart backend
```

#### 3. 前端无法访问后端
```bash
# 检查网络连接
docker network ls
docker network inspect sign-language-learning_signavatar-network

# 检查环境变量
docker-compose exec frontend env | grep REACT_APP
```

#### 4. SSL证书问题
```bash
# 检查证书有效性
openssl x509 -in nginx/ssl/cert.pem -text -noout

# 重新生成证书
rm nginx/ssl/*
# 重新执行SSL配置步骤
```

### 性能调优

#### 数据库优化
```bash
# 进入后端容器
docker-compose exec backend bash

# 运行数据库优化
python -m backend.scripts.optimize_db
```

#### 缓存配置
```bash
# 调整Redis内存限制
# 编辑 docker-compose.yml 中的 Redis 配置
command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## 升级指南

### 1. 备份数据
```bash
./backup.sh
```

### 2. 更新代码
```bash
git pull origin main
```

### 3. 重新部署
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 4. 运行迁移
```bash
docker-compose exec backend python -m backend.scripts.migrate
```

## 安全建议

1. **定期更新**: 保持系统和依赖包的最新版本
2. **强密码**: 使用复杂的密码和密钥
3. **防火墙**: 只开放必要的端口
4. **SSL/TLS**: 在生产环境中使用HTTPS
5. **备份**: 定期备份重要数据
6. **监控**: 设置系统监控和告警
7. **日志**: 定期检查和分析日志文件

## 技术支持

如果遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查项目的 [Issues](https://github.com/your-org/sign-language-learning/issues)
3. 提交新的 Issue 并提供详细的错误信息
4. 联系技术支持团队
