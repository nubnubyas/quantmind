#!/bin/bash
# QuantMind 测试运行器（自动检测基础设施状态）
# Usage: bash scripts/run-tests.sh [pytest args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# 检查基础设施
PG_UP=false
QD_UP=false

if docker compose -f docker-compose.yml ps postgres 2>/dev/null | grep -q "Up\|running"; then
    PG_UP=true
fi

if docker compose -f docker-compose.yml ps qdrant 2>/dev/null | grep -q "Up\|running"; then
    QD_UP=true
fi

echo "=== QuantMind 测试 ==="
echo "PostgreSQL: $([ "$PG_UP" = true ] && echo '✅ 运行中' || echo '⚠️  未启动 (DB 测试将跳过)')"
echo "Qdrant:     $([ "$QD_UP" = true ] && echo '✅ 运行中' || echo '⚠️  未启动 (Qdrant 测试将跳过)')"
echo ""

export POSTGRES_URL="${POSTGRES_URL:-postgresql://quantmind:quantmind@localhost:5432/quantmind}"
export QDRANT_HOST="${QDRANT_HOST:-localhost}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"

pytest tests/ -v "$@"
