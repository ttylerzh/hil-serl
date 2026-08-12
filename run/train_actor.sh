XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  PYTHONPATH=.:examples \
  python examples/train_rlpd.py \
    --exp_name=plug_insertion \
    --checkpoint_path=rl_ckpt/insert_plug \
    --actor
