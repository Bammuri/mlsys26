#!/bin/bash
# Hook: On session start, show current optimization status
# Reads latest benchmark results and shows summary

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_LOG="$PROJECT_ROOT/logs/bench_history.jsonl"

if [ -f "$BENCH_LOG" ] && [ -s "$BENCH_LOG" ]; then
    echo "=== GDN Optimization Status ==="

    # Last decode result
    LAST_DECODE=$(grep '"decode"' "$BENCH_LOG" | tail -1)
    if [ -n "$LAST_DECODE" ]; then
        DECODE_SPEEDUP=$(echo "$LAST_DECODE" | jq -r '.mean_speedup // "N/A"')
        DECODE_STATUS=$(echo "$LAST_DECODE" | jq -r '.status // "N/A"')
        DECODE_DATE=$(echo "$LAST_DECODE" | jq -r '.timestamp // "N/A"' | cut -c1-10)
        echo "Decode: ${DECODE_SPEEDUP}x speedup (${DECODE_STATUS}, ${DECODE_DATE})"
    else
        echo "Decode: no benchmark yet"
    fi

    # Last prefill result
    LAST_PREFILL=$(grep '"prefill"' "$BENCH_LOG" | tail -1)
    if [ -n "$LAST_PREFILL" ]; then
        PREFILL_SPEEDUP=$(echo "$LAST_PREFILL" | jq -r '.mean_speedup // "N/A"')
        PREFILL_STATUS=$(echo "$LAST_PREFILL" | jq -r '.status // "N/A"')
        PREFILL_DATE=$(echo "$LAST_PREFILL" | jq -r '.timestamp // "N/A"' | cut -c1-10)
        echo "Prefill: ${PREFILL_SPEEDUP}x speedup (${PREFILL_STATUS}, ${PREFILL_DATE})"
    else
        echo "Prefill: no benchmark yet"
    fi

    TOTAL=$(wc -l < "$BENCH_LOG")
    echo "Total benchmark runs: $TOTAL"
    echo "==="
else
    echo "=== GDN Optimization: No benchmarks yet. Use /bench to run first benchmark. ==="
fi

exit 0
