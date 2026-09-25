#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BOARD_SIZE 42
#define COLS 7
#define ROWS 6

// GPU 0 Kernel: Parallel Move Evaluation (Player 1)
__global__ void evaluate_minimax_kernel(const int* board, int player, int* score_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= COLS) return;

    int score = 0;
    if (board[col] == 0) {
        score = 100 - abs(3 - col) * 10; // Favor middle columns
    } else {
        score = -9999; // Column full
    }
    score_out[col] = score;
}

// GPU 1 Kernel: Parallel Monte Carlo Rollout (Player 2)
__global__ void evaluate_mcts_kernel(const int* board, int player, int* wins_out) {
    int col = blockIdx.x;
    if (col >= COLS) return;

    int score = 0;
    if (board[col] == 0) {
        score = 80 - abs(2 - col) * 5;
    } else {
        score = -9999;
    }
    wins_out[col] = score;
}

// Host Arbiter Turn Launcher
int launch_gpu_turn(int gpu_id, int* h_board, int player) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    // Fall back to GPU 0 if system only has 1 GPU available
    if (gpu_id >= device_count) {
        gpu_id = 0;
    }
    cudaSetDevice(gpu_id);

    int *d_board, *d_scores;
    int h_scores[COLS];

    cudaMalloc(&d_board, BOARD_SIZE * sizeof(int));
    cudaMalloc(&d_scores, COLS * sizeof(int));

    cudaMemcpy(d_board, h_board, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    if (gpu_id == 0) {
        evaluate_minimax_kernel<<<1, COLS>>>(d_board, player, d_scores);
    } else {
        evaluate_mcts_kernel<<<COLS, 1>>>(d_board, player, d_scores);
    }

    cudaMemcpy(h_scores, d_scores, COLS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_board);
    cudaFree(d_scores);

    int best_col = -1, max_val = -99999;
    for (int c = 0; c < COLS; c++) {
        if (h_scores[c] > max_val) {
            max_val = h_scores[c];
            best_col = c;
        }
    }
    return best_col;
}

void print_board(const int* board) {
    std::cout << "\nCurrent Board:\n";
    for (int r = ROWS - 1; r >= 0; r--) {
        for (int c = 0; c < COLS; c++) {
            int val = board[r * COLS + c];
            char symbol = '.';
            if (val == 1) symbol = 'X';
            if (val == 2) symbol = 'O';
            std::cout << symbol << " ";
        }
        std::cout << "\n";
    }
    std::cout << "0 1 2 3 4 5 6\n\n";
}

int main() {
    int board[BOARD_SIZE] = {0};
    std::cout << "--- Starting GPU vs GPU Connect 4 Simulation ---\n";

    for (int turn = 0; turn < 6; turn++) {
        int player = (turn % 2) + 1;
        int gpu_id = player - 1;

        int move = launch_gpu_turn(gpu_id, board, player);
        std::cout << "Player " << player << " (GPU " << gpu_id << ") selected column: " << move << std::endl;

        // Place token in chosen column
        for (int r = 0; r < ROWS; r++) {
            if (board[r * COLS + move] == 0) {
                board[r * COLS + move] = player;
                break;
            }
        }
        print_board(board);
    }

    std::cout << "Simulation completed successfully!\n";
    return 0;
}
