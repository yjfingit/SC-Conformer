#!/bin/bash
pkill -f "run_all.py" 2>/dev/null
pkill -f "train.py --dataset" 2>/dev/null
pkill -f "chain_ea" 2>/dev/null
pkill -f "chain_cbra" 2>/dev/null
pkill -f "launch_seed4" 2>/dev/null
sleep 2
echo "remaining training procs:"
pgrep -fc "train.py --dataset" || echo 0
free -g | head -2
