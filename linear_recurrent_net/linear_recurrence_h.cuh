#ifndef RECURR_H_
#define RECURR_H_

void gpuAssert(cudaError_t code, const char *file, int line);
__device__ int2 divide_work(int n_jobs, int n_workers, int worker_idx);
__device__ int2 compute_warp_start_stop(int block_idx, int warp_idx, int n_blocks, int n_steps);

#endif 