#!/bin/bash
cd /root/autodl-tmp/ICLR
# wait for the seed grid to finish, then run cbramod comparison on seed + seed4
NSEED=$(find results/seed -name "*.json" 2>/dev/null | wc -l)
while [ "$NSEED" -lt 90 ]; do
    NSEED=$(find results/seed -name "*.json" 2>/dev/null | wc -l)
    echo "$(date +%H:%M) seed=$NSEED/90"
    sleep 300
done
echo "seed done -> cbramod comparison"
python run_all.py --datasets seed seed4 --models cbramod --epochs 20 --batch 128 --parallel 1 >> /tmp/gcbra.log 2>&1
echo "cbramod done $(date +%H:%M)"
