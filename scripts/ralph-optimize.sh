#!/bin/bash
# ralph-optimize.sh — Iterative GDN kernel optimization loop
#
# Usage:
#   ./scripts/ralph-optimize.sh [target] [max_iter] [budget_per_iter]
#   ./scripts/ralph-optimize.sh prefill 10 5.00
#   ./scripts/ralph-optimize.sh decode 5
#   ./scripts/ralph-optimize.sh both 8

set -euo pipefail

TARGET="${1:-prefill}"
MAX_ITER="${2:-10}"
BUDGET="${3:-5.00}"

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Validate target
if [[ "$TARGET" != "decode" && "$TARGET" != "prefill" && "$TARGET" != "both" ]]; then
    echo "Error: target must be decode, prefill, or both"
    exit 1
fi

# Build target list
if [[ "$TARGET" == "both" ]]; then
    TARGETS=(prefill decode)
else
    TARGETS=("$TARGET")
fi

LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"/{decode,prefill}

echo "============================================"
echo "  GDN Ralph Optimization Loop"
echo "  Targets: ${TARGETS[*]}"
echo "  Max iterations: $MAX_ITER"
echo "  Budget per iteration: \$$BUDGET"
echo "  Started: $(date -Iseconds)"
echo "============================================"
echo ""

for i in $(seq 1 "$MAX_ITER"); do
    for kern in "${TARGETS[@]}"; do
        ITER_LOG="$LOG_DIR/$kern/iter_${i}.log"
        HISTORY="$LOG_DIR/$kern/bench_history.jsonl"

        # Get baseline from last non-reverted entry
        BEFORE_SPEEDUP="none"
        BEFORE_LATENCY="none"
        if [[ -f "$HISTORY" ]]; then
            read -r BEFORE_SPEEDUP BEFORE_LATENCY < <(python3 -c "
import json
last_good = None
for line in open('$HISTORY'):
    d = json.loads(line)
    if not d.get('reverted', False):
        last_good = d
if last_good:
    print(last_good.get('mean_speedup','?'), last_good.get('mean_latency_ms','?'))
else:
    print('?', '?')
" 2>/dev/null || echo "? ?")
        fi

        echo "=== [$kern] Iteration $i/$MAX_ITER (current: ${BEFORE_SPEEDUP}x, ${BEFORE_LATENCY}ms) ==="
        echo "    Start: $(date -Iseconds)"

        # Snapshot kernel before change for rollback
        KERNEL_DIR="gdn_${kern}_qk4_v8_d128_k_last/solution/cuda"
        cp "$KERNEL_DIR/kernel.cu" "$KERNEL_DIR/kernel.cu.bak"

        # Run optimization iteration
        if claude -p \
            --dangerously-skip-permissions \
            --model claude-opus-4-6 \
            --max-budget-usd "$BUDGET" \
            --effort high \
            "/optimize $kern" \
            2>&1 | tee "$ITER_LOG"; then
            RUN_OK=true
        else
            RUN_OK=false
        fi

        # Check result (last entry = this iteration's benchmark)
        if [[ -f "$HISTORY" ]]; then
            LAST=$(tail -1 "$HISTORY")
            read -r STATUS AFTER_SPEEDUP AFTER_LATENCY < <(echo "$LAST" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('status','unknown'), d.get('mean_speedup','?'), d.get('mean_latency_ms','?'))
" 2>/dev/null || echo "unknown ? ?")
        else
            STATUS="unknown"
            AFTER_SPEEDUP="?"
            AFTER_LATENCY="?"
        fi

        if [[ "$STATUS" != "pass" || "$RUN_OK" != "true" ]]; then
            echo "    FAIL (status=$STATUS) — rolling back kernel"
            cp "$KERNEL_DIR/kernel.cu.bak" "$KERNEL_DIR/kernel.cu"
        else
            echo "    PASS: ${BEFORE_SPEEDUP}x -> ${AFTER_SPEEDUP}x (latency: ${BEFORE_LATENCY}ms -> ${AFTER_LATENCY}ms)"
        fi

        # Clean up backup
        rm -f "$KERNEL_DIR/kernel.cu.bak"

        echo "    End: $(date -Iseconds)"
        echo ""
    done
done

echo "============================================"
echo "  Loop complete: $(date -Iseconds)"
echo "============================================"
echo ""
echo "Final results:"
for kern in "${TARGETS[@]}"; do
    HISTORY="$LOG_DIR/$kern/bench_history.jsonl"
    if [[ -f "$HISTORY" ]]; then
        echo "  [$kern]"
        echo "    Entries: $(wc -l < "$HISTORY")"
        echo "    Latest: $(tail -1 "$HISTORY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"mean_speedup\"]}x ({d[\"status\"]})')" 2>/dev/null || echo "?")"
        echo "    History:"
        python3 -c "
import json, sys
for line in open('$HISTORY'):
    d = json.loads(line)
    ts = d.get('timestamp','?')[:16]
    sp = d.get('mean_speedup','?')
    st = d.get('status','?')
    n = d.get('notes','')[:60]
    print(f'      {ts}  {sp:>8}x  {st:>4}  {n}')
" 2>/dev/null || echo "      (parse error)"
    fi
done
