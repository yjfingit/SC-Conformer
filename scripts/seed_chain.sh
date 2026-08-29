#!/bin/bash
# Chain: wait until the other grids release RAM, then run SEED grid.
cd /root/autodl-tmp/ICLR
SEED_JSON=$(find results/seed -name "*.json" 2>/dev/null | wc -l)
while true; do
    MEM=$(cat /sys/fs/cgroup/memory.current)
    N2A=$(find results/bnci_2a -name "*.json" 2>/dev/null | wc -l)
    N2B=$(find results/bnci_2b -name "*.json" 2>/dev/null | wc -l)
    N002=$(find results/bnci_2014_002 -name "*.json" 2>/dev/null | wc -l)
    echo "$(date +%H:%M) mem=$(($MEM/1024/1024/1024))GB 2a=$N2A 2b=$N2B 002=$N002 seed=$SEED_JSON"
    if [ "$N2A" -ge 110 ] && [ "$N2B" -ge 100 ] && [ "$N002" -ge 160 ]; then
        echo "grids done -> launching SEED grid"
        break
    fi
    if [ "$MEM" -lt 40000000000 ]; then
        RUNNOW=1
    fi
    sleep 300
done
python run_all.py --datasets seed --models eegnet conformer atcnet scformer scformer-v2 shallow --epochs 20 --batch 256 --parallel 2
