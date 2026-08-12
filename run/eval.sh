PYTHONPATH=.:examples \
  python examples/train_rlpd.py \
    --exp_name=pick \
    --checkpoint_path=rl_ckpt/pick_0805 \
    --actor \
    --eval_checkpoint_step=5000 \
    --eval_n_trajs=20
