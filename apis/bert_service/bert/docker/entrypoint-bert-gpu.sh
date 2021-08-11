#!/bin/sh
bert-serving-start \
  -model_dir /model \
  -max_seq_len=NONE \
  -num_worker=${NUM_WORKER:-1}
