# GPU-vs.-GPU-Game-Architecture-Connect-4
A dual-GPU Connect 4 engine where competitors use distinct CUDA strategies. GPU 0 runs a parallel Minimax search across blocks to evaluate board trees, while GPU 1 uses Monte Carlo rollouts across threads to calculate win probabilities. The host manages turns, state synchronization, and `cudaSetDevice()` transfers.
