#!/bin/bash
pkill -f "run_all.py --datasets seed" 2>/dev/null
pkill -f "train.py --dataset seed" 2>/dev/null
sleep 2
cd /root/autodl-tmp/ICLR
mkdir -p results_noea
mv results/bnci_2a results_noea/ 2>/dev/null
mv results/bnci_2b results_noea/ 2>/dev/null
mv results/seed results_noea/ 2>/dev/null
ls results/
python src/data/attach_ea.py bnci_2a bnci_2b seed
