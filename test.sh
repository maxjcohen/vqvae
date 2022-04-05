#!/bin/sh
EPOCHS=20
LR=1e-3
NUM_WOKERS=4
GPUS=1

for flavor in classic ema gumbel; do
    python -m scripts.dataset.cifar \
	--gpus $GPUS \
	--num-workers $NUM_WOKERS \
	--epochs $EPOCHS \
	--lr $LR \
	--flavor $flavor
done
