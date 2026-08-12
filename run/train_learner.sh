XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.3 \
  PYTHONPATH=.:examples \
  python examples/train_rlpd.py \
    --exp_name=plug_insertion \
    --checkpoint_path=rl_ckpt/insert_plug\
    --demo_path=demo_data/insert_plug_30_demos_2026-08-12_16-32-13.pkl \
    --learner \
    --debug
