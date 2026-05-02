from train_maskable_ppo import parse_args
args = parse_args()
max_steps = max(1, int(min(args.episode_len * args.bc_trace_fraction, args.bc_max_samples)))
print(f"args.episode_len={args.episode_len}")
print(f"args.bc_trace_fraction={args.bc_trace_fraction}")
print(f"max_steps={max_steps}")
