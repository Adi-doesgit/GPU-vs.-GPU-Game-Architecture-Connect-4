System Design & Multi-GPU StrategyThis design implements a competitive 2-GPU Connect 4 game where two CUDA kernels execute distinct AI decision strategies.       
      
      [ Host Arbiter / Game Loop ]
         /                    \
      (Device 0)                (Device 1)
    [GPU 1: Parallel Minimax]  [GPU 2: Monte Carlo Rollout]
    Evaluate 7 branches        Simulate N random games
        \                      /
         [ Output Move Selection ]

         
* Game State Representation: A 1D array of size 42 (int board[42]), where 0 = empty, 1 = Player 1, 2 = Player 2.

* Turn Coordination: The host process manages the master game board, alternates turns, and uses cudaSetDevice() to dispatch state data to the active GPU. For single-GPU hardware, two host threads run separate executable processes using file locks (p1.lock / p2.lock) to read/write state from shared storage.
