#!/bin/sh
# Prometheus (127.0.0.1:9090) and Grafana (127.0.0.1:3000, dashboard "mynah-asr")
# on loopback, scraping a server started with --metrics-port 9101 (CPU) and/or
# 9102 (GPU). Validates the configs first when promtool is available.
#
# Usage: configs/observability/observability_up.sh [up|down] [--otel]
set -e
cd "$(dirname "$0")"
cmd="${1:-up}"
profile=""
[ "$2" = "--otel" ] && profile="--profile otel"
command -v docker >/dev/null 2>&1 || { echo "observability_up: docker is required"; exit 1; }
if [ "$cmd" = down ]; then
    docker compose $profile down
    exit 0
fi
if command -v promtool >/dev/null 2>&1; then
    promtool check rules alerts.yml
fi
for port in 9101 9102; do
    if curl -sf "http://127.0.0.1:$port/metrics" >/dev/null 2>&1; then
        echo "observability_up: a server answers /metrics on 127.0.0.1:$port"
    else
        echo "observability_up: nothing on 127.0.0.1:$port yet (start the server with --metrics-port $port)"
    fi
done
docker compose $profile up -d
echo "Prometheus http://127.0.0.1:9090   Grafana http://127.0.0.1:3000 (dashboard: mynah/mynah-asr)"
