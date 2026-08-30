#!/bin/bash
# Full matrix + sweep passes to fill any folds lost to transient OOM.
cd /root/autodl-tmp/ICLR
MODELS="eegnet-de deep4-de personal-zscore riemannian-mdm tsception ppda-zs ppda-uda scde emT mshcl ama-eeg"
python run_all_de.py $MODELS --parallel 4 >> /tmp/gmatrix.log 2>&1
# sweep pass 1
python run_all_de.py $MODELS --parallel 3 >> /tmp/gmatrix_sweep1.log 2>&1
# sweep pass 2
python run_all_de.py $MODELS --parallel 2 >> /tmp/gmatrix_sweep2.log 2>&1
echo "ALL DONE $(date +%H:%M)"
for m in $MODELS majority; do
    echo "$m: $(find results_de/$m -name '*.json' | wc -l)/15"
done
