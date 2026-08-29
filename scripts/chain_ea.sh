#!/bin/bash
cd /root/autodl-tmp/ICLR
# wait for 2a/2b grids to finish (free RAM)
while true; do
    N2A=$(find results/bnci_2a -name "*.json" 2>/dev/null | wc -l)
    N2B=$(find results/bnci_2b -name "*.json" 2>/dev/null | wc -l)
    MEM=$(cat /sys/fs/cgroup/memory.current)
    echo "$(date +%H:%M) 2a=$N2A 2b=$N2B mem=$(($MEM/1024/1024/1024))GB"
    if [ "$N2A" -ge 108 ] && [ "$N2B" -ge 108 ]; then break; fi
    sleep 180
done
python run_all.py --datasets seed --models eegnet conformer atcnet scformer scformer-v2 shallow --epochs 20 --batch 256 --parallel 2 >> /tmp/gseed.log 2>&1
echo "seed grid done $(date +%H:%M)"
python run_all.py --datasets seed4 --models eegnet conformer atcnet scformer scformer-v2 shallow --epochs 20 --batch 256 --parallel 2 >> /tmp/gseed4.log 2>&1
echo "seed4 grid done $(date +%H:%M)"
