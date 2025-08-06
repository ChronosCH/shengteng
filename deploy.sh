#!/bin/bash

# SignAvatar Web 自动化部署脚本
# 支持开发和生产环境部署

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

print_success() {
    print_message "✅ $1" $GREEN
}

print_error() {
    print_message "❌ $1" $RED
}

print_warning() {
    print_message "⚠️ $1" $YELLOW
}

print_info() {
    print_message "ℹ️ $1" $BLUE
}

# 检查依赖
check_dependencies() {
    print_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装。请先安装 Docker。"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装。请先安装 Docker Compose。"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 未安装。请先安装 Python 3.9+。"
        exit 1
    fi
    
    print_success "所有依赖检查通过"
}

# 环境设置
setup_environment() {
    print_info "设置环境配置..."
    
    # 复制环境配置文件
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "创建 .env 配置文件"
        else
            print_error ".env.example 文件不存在"
            exit 1
        fi
    fi
    
    # 创建必要的目录
    mkdir -p data logs uploads models temp monitoring/grafana/{dashboards,datasources} nginx/ssl
    
    # 设置权限
    chmod +x start.py
    chmod +x deploy.sh
    
    print_success "环境设置完成"
}

# 开发环境部署
deploy_development() {
    print_info "部署开发环境..."
    
    # 安装Python依赖
    print_info "安装Python依赖..."
    pip3 install -r requirements.txt
    
    # 初始化项目
    print_info "初始化项目..."
    python3 start.py init
    
    # 启动开发服务器
    print_info "启动开发服务器..."
    python3 start.py start --reload &
    
    # 等待服务器启动
    sleep 5
    
    # 健康检查
    python3 start.py check-health
    
    print_success "开发环境部署完成"
    print_info "API文档: http://localhost:8000/api/docs"
    print_info "停止服务: Ctrl+C"
}

# 生产环境部署
deploy_production() {
    print_info "部署生产环境..."
    
    # 拉取最新镜像
    print_info "拉取依赖镜像..."
    docker-compose pull redis prometheus grafana nginx
    
    # 构建应用镜像
    print_info "构建应用镜像..."
    docker-compose build
    
    # 启动服务
    print_info "启动生产服务..."
    docker-compose up -d
    
    # 等待服务启动
    print_info "等待服务启动..."
    sleep 30
    
    # 健康检查
    print_info "进行健康检查..."
    for i in {1..10}; do
        if curl -f http://localhost:8000/api/health &>/dev/null; then
            print_success "服务健康检查通过"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "服务启动失败"
            docker-compose logs
            exit 1
        fi
        sleep 5
    done
    
    print_success "生产环境部署完成"
    print_info "应用地址: http://localhost"
    print_info "API文档: http://localhost/api/docs"
    print_info "监控面板: http://localhost:3001 (admin/admin)"
    print_info "Prometheus: http://localhost:9090"
}

# 更新部署
update_deployment() {
    print_info "更新部署..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        print_info "更新开发环境..."
        # 重启开发服务器
        pkill -f "uvicorn"
        python3 start.py start --reload &
    else
        print_info "更新生产环境..."
        # 重新构建并重启服务
        docker-compose build
        docker-compose up -d --force-recreate
    fi
    
    print_success "更新完成"
}

# 停止服务
stop_services() {
    print_info "停止服务..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        pkill -f "uvicorn" || true
        print_success "开发服务器已停止"
    else
        docker-compose down
        print_success "生产服务已停止"
    fi
}

# 清理环境
cleanup() {
    print_info "清理环境..."
    
    stop_services
    
    if [ "$ENVIRONMENT" = "production" ]; then
        # 清理Docker资源
        docker-compose down --volumes --remove-orphans
        docker system prune -f
    fi
    
    # 清理临时文件
    rm -rf temp/* logs/*.log
    
    print_success "环境清理完成"
}

# 备份数据
backup_data() {
    print_info "备份数据..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 备份数据库
    if [ -f "data/signavatar.db" ]; then
        cp data/signavatar.db "$BACKUP_DIR/"
        print_success "数据库备份完成"
    fi
    
    # 备份上传文件
    if [ -d "uploads" ]; then
        cp -r uploads "$BACKUP_DIR/"
        print_success "上传文件备份完成"
    fi
    
    # 备份配置文件
    cp .env "$BACKUP_DIR/" 2>/dev/null || true
    
    print_success "数据备份完成: $BACKUP_DIR"
}

# 显示日志
show_logs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        tail -f logs/*.log 2>/dev/null || echo "没有找到日志文件"
    else
        docker-compose logs -f
    fi
}

# 显示状态
show_status() {
    print_info "系统状态："
    
    if [ "$ENVIRONMENT" = "development" ]; then
        if pgrep -f "uvicorn" &>/dev/null; then
            print_success "开发服务器运行中"
        else
            print_warning "开发服务器未运行"
        fi
    else
        docker-compose ps
    fi
    
    # 检查端口
    print_info "\n端口状态："
    netstat -tuln | grep -E ':(8000|3000|6379|9090|3001|80|443)' || print_warning "未找到监听端口"
}

# 运行测试
run_tests() {
    print_info "运行测试套件..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        python3 test_system.py
        python3 start.py test
    else
        docker-compose exec backend python test_system.py
    fi
}

# 显示帮助
show_help() {
    echo "SignAvatar Web 部署脚本"
    echo ""
    echo "用法: $0 [选项] <命令>"
    echo ""
    echo "选项:"
    echo "  -e, --environment ENV    设置环境 (development|production)"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "命令:"
    echo "  deploy                  部署应用"
    echo "  update                  更新部署"
    echo "  stop                    停止服务"
    echo "  restart                 重启服务"
    echo "  cleanup                 清理环境"
    echo "  backup                  备份数据"
    echo "  logs                    显示日志"
    echo "  status                  显示状态"
    echo "  test                    运行测试"
    echo ""
    echo "示例:"
    echo "  $0 -e development deploy     # 部署开发环境"
    echo "  $0 -e production deploy      # 部署生产环境"
    echo "  $0 status                    # 显示状态"
    echo "  $0 logs                      # 显示日志"
}

# 主函数
main() {
    # 默认环境
    ENVIRONMENT="development"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy)
                COMMAND="deploy"
                shift
                ;;
            update)
                COMMAND="update"
                shift
                ;;
            stop)
                COMMAND="stop"
                shift
                ;;
            restart)
                COMMAND="restart"
                shift
                ;;
            cleanup)
                COMMAND="cleanup"
                shift
                ;;
            backup)
                COMMAND="backup"
                shift
                ;;
            logs)
                COMMAND="logs"
                shift
                ;;
            status)
                COMMAND="status"
                shift
                ;;
            test)
                COMMAND="test"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证环境参数
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
        print_error "无效的环境: $ENVIRONMENT"
        exit 1
    fi
    
    print_info "环境: $ENVIRONMENT"
    
    # 执行命令
    case $COMMAND in
        deploy)
            check_dependencies
            setup_environment
            if [ "$ENVIRONMENT" = "development" ]; then
                deploy_development
            else
                deploy_production
            fi
            ;;
        update)
            update_deployment
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            if [ "$ENVIRONMENT" = "development" ]; then
                deploy_development
            else
                deploy_production
            fi
            ;;
        cleanup)
            cleanup
            ;;
        backup)
            backup_data
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        test)
            run_tests
            ;;
        *)
            print_error "请指定命令"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
