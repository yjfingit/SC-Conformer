#!/bin/bash
cd /root/autodl-tmp/ICLR
pkill -f "train.py --dataset seed4" 2>/dev/null
pkill -f "gseed4" 2>/dev/null
sleep 2
rm -rf results/seed4
exec python run_all.py --datasets seed4 --models eegnet conformer atcnet scformer scformer-v2 shallow --epochs 20 --batch 256 --parallel 2
