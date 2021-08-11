#!/bin/sh
bert-serving-start \
  -cpu -max_batch_size 16 \
  -max_seq_len=NONE \
  -model_dir /model \
  -num_worker=${NUM_WORKER:-1}
