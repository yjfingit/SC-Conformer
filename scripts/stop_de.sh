#!/bin/bash
pkill -f "train_de.py" 2>/dev/null
pkill -f "run_all_de.py" 2>/dev/null
sleep 2
echo "remaining: $(pgrep -fc 'train_de.py' || echo 0)"
