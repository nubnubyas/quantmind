#!/bin/bash
# QuantMind 开发环境一键启动
# Usage: bash scripts/dev-setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== QuantMind 开发环境启动 ==="

# 1. 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装或不在 PATH 中"
    exit 1
fi

# 2. 启动基础设施容器
echo "→ 启动 Qdrant + PostgreSQL..."
docker compose -f docker-compose.yml up -d postgres qdrant

# 3. 等待 PostgreSQL 就绪
echo "→ 等待 PostgreSQL 就绪..."
ATTEMPTS=0
MAX_ATTEMPTS=30
until docker compose -f docker-compose.yml exec -T postgres pg_isready -U quantmind &> /dev/null; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo "❌ PostgreSQL 启动超时"
        exit 1
    fi
    sleep 1
done
echo "✅ PostgreSQL 就绪"

# 4. 检查 .env
if [ ! -f .env ]; then
    echo "⚠️  .env 文件不存在，从 .env.example 复制"
    cp .env.example .env
    echo "请编辑 .env 填入 DEEPSEEK_API_KEY 后重新运行"
    exit 1
fi

# 5. 激活虚拟环境（可选）
if [ -d .venv ]; then
    echo "→ 激活 .venv"
    source .venv/bin/activate
fi

echo ""
echo "=== 开发环境就绪 ==="
echo "基础设施: docker compose -f docker-compose.yml ps"
echo "启动 API:  uvicorn src.api.main:app --reload --port 8000"
echo "运行测试:  pytest tests/ -v"
echo "启动 UI:   streamlit run src/ui/streamlit_app.py"
