#include <assert.h>
#include <stdio.h>
// #include "exclusiveScan.cu_inl"

#define SCAN_ARRS_PER_BLK 24
#define SCAN_BLOCK_DIM 256

#define CEIL_DIV(x, y) ((x + y - 1) / y)

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

__inline__ __device__ void sharedMemExclusiveScan(float* decays, float* impulses, int size);

__device__ int2 divide_work(int n_jobs, int n_workers, int worker_idx);

__device__ int2 compute_warp_start_stop(int block_idx, int warp_idx,
          int n_blocks, int n_steps);

// decay storage, h_storage:
//   each a n_dims x 33 x n_blocks matrix on GPU with 33rd column for block reduction
__global__ void reduction_kernel_fast(float *decays, float *impulses,
         float *initial_state,
         float *_decay_storage, float *_h_storage,
         int n_dims, int n_steps) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  float *decay_storage = &_decay_storage[blockIdx.x * 33 * n_dims];
  float *h_storage = &_h_storage[blockIdx.x * 33 * n_dims];

  int2 start_stop = compute_warp_start_stop(blockIdx.x, warp, gridDim.x, n_steps);
  int warp_start = start_stop.x;
  int warp_stop = start_stop.y;

  /*
  * Reduce within warps.
  * After this loop exits, the storage arrays should contain the reduction
  * from warp_start to warp_stop (including initial state) at index
  * (feature_idx, warp, block).
  */
  for (int i = lane; i < n_dims; i += 32) {
    float cum_decay = 1.0;
    float h = 0.0;
    if (blockIdx.x == 0 && warp == 0 && initial_state != NULL) {
      h = initial_state[i];
    }

    for (int t = warp_start; t < warp_stop; t++) {
      cum_decay *= decays[i + t * n_dims];
      h = decays[i + t * n_dims] * h + impulses[i + t * n_dims];
    }

    // TODO: store into shared memory, work in shared memory sized blocks
    // store into global memory
    decay_storage[i + warp * n_dims] = cum_decay;
    h_storage[i + warp * n_dims] = h;
  }

  __syncthreads();

  /*
   * Reduce over warps.
   * After this loop exits, the storage arrays should contain the reduction
   * from block_start to block_finish (including initial state) at index
   * (feature_idx, 32, block).
   */
  // TODO: parallel reduction (or scan). Need to worry about changing the warp
  //       reduction values (as I use them again later)
  for (int i = lane + 32 * warp; i < n_dims; i += blockDim.x) {
    float cum_decay = 1.0;
    float h = 0.0;
    for (int t = 0; t < 32; t++) {
      cum_decay *= decay_storage[i + t * n_dims];
      h = decay_storage[i + t * n_dims] * h + h_storage[i + t * n_dims];
    }
    decay_storage[i + 32 * n_dims] = cum_decay;
    h_storage[i + 32 * n_dims] = h;
  }
}

__global__ void block_scan_kernel_fast(float *decay_storage, float *h_storage,
          int n_dims, int n_reduced_blocks) {
  /*
   * Scan over blocks.
   * After this loop exits, the storage arrays should contain the cumulative sum
   * from block_idx 0 to i (inclusive) at index (feature_idx, 32, i)
   * This means (feature_idx, 32, 2) contains the reduction of blocks 0, 1, and 2.
   */
  __shared__ float decay_arrays[SCAN_ARRS_PER_BLK * SCAN_BLOCK_DIM];
  __shared__ float impulse_arrays[SCAN_ARRS_PER_BLK * SCAN_BLOCK_DIM];

  int n_arrs = min(SCAN_ARRS_PER_BLK, n_dims - blockIdx.x * SCAN_ARRS_PER_BLK);
  int storage_offset = threadIdx.x * n_dims + blockIdx.x * SCAN_ARRS_PER_BLK;

  // Cooperatively load arrays
  if (threadIdx.x < n_reduced_blocks) {
    for (int i = 0; i < n_arrs; i++) {
      int storage_idx = storage_offset + i;
      int array_idx = i * SCAN_BLOCK_DIM + threadIdx.x;
      decay_arrays[array_idx] = decay_storage[storage_idx];
      impulse_arrays[array_idx] = h_storage[storage_idx];
    }
  }

  __syncthreads();

  // Scan each array
  for (int which_array = 0; which_array < n_arrs; which_array++) {
    int array_start = which_array * SCAN_BLOCK_DIM;
    float *decays_start = &decay_arrays[array_start];
    float *impulses_start = &impulse_arrays[array_start];
    sharedMemExclusiveScan(decays_start, impulses_start, SCAN_BLOCK_DIM);
  }

  __syncthreads();

  // Cooperatively store arrays
  if (threadIdx.x < n_reduced_blocks) {
    for (int i = 0; i < n_arrs; i++) {
      int storage_idx = storage_offset + i;
      int array_idx = i * n_reduced_blocks + threadIdx.x;
      decay_storage[storage_idx] = decay_arrays[array_idx];
      h_storage[storage_idx] = impulse_arrays[array_idx];
    }
  }

}

__global__ void warp_scan_kernel_fast(float *decays, float *impulses,
         float *initial_state, float *out,
         float *decay_storage, float *h_storage,
         int n_dims, int n_steps) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  // Note: Due to the index ordering of the storage arrays, the following
  // indices are equivalent:
  //
  // i + (t - 1) * n_dims + blockIdx.x * 33 * n_dims
  // i + 32 * n_dims + (blockIdx.x - 1) * 33 * n_dims
  //
  // when t is 0. This means something that looks like negative indexing
  // (t-1) can be used to safely access the stored value for the previous
  // warp (even if the previous warp belonged to the previous block).

  /*
   * Scan over warps.
   * After this loop executes, the storage arrays should contain the cumulative
   * sum from the beginning of sequence (including initial condition) up to
   * and including the indexed warp and block.
   */
  // TODO: parallel scan
  for (int i = lane + 32 * warp; i < n_dims; i += blockDim.x) {
    for (int t = 0; t < 32; t++) {
      if (t == 0 && blockIdx.x == 0) {
        // the reduction over warp 0 (including initial condition) is correct val
        // for scan, so there's no work to do
        continue;
      }

      int cur_idx = i + t * n_dims + blockIdx.x * 33 * n_dims;
      int prev_idx = i + (t - 1) * n_dims + blockIdx.x * 33 * n_dims;
      h_storage[cur_idx] = decay_storage[cur_idx] * h_storage[prev_idx] + h_storage[cur_idx];
      decay_storage[cur_idx] *= decay_storage[prev_idx];
    }
  }

  __syncthreads();

  int2 start_stop = compute_warp_start_stop(blockIdx.x, warp, gridDim.x, n_steps);
  int warp_start = start_stop.x;
  int warp_stop = start_stop.y;

  /*
   * Scan within warps.
   * This loop writes to the output array. Each warp reads in it's initial state
   * (either from the "initial_state" or the storage arrays) and then writes
   * to output for indices warp_start up to warp_stop.
   */
  for (int i = lane; i < n_dims; i += 32) {
    float h = 0.0;
    if (blockIdx.x == 0 && warp == 0) {
      if (initial_state != NULL) {
  h = initial_state[i];
      }
    } else {
      h = h_storage[i + (warp - 1) * n_dims + blockIdx.x * 33 * n_dims];
    }

    for (int t = warp_start; t < warp_stop; t++) {
      h = decays[i + t * n_dims] * h + impulses[i + t * n_dims];
      out[i + t * n_dims] = h;
    }
  }
}

extern "C" {
/*
 * This is the main method for the prefix sum kernels.
 * decays, impulses, out:
 *   each a n_dims x n_steps column major matrix located on GPU
 * initial_state:
 *   array of size n_dims located on GPU
 */
void compute_fast_linear_recurrence(float *decays, float *impulses, float *initial_state,
             float *out, int n_dims, int n_steps) {

  // TODO: query
  int n_SMs = 15;
  int n_blocks_per_sm = 16;

  // we want at least 32 elements per block, but no reason to run
  // with more than the maximum number of concurrent blocks
  int n_blocks = min(CEIL_DIV(n_steps, 32), n_SMs * n_blocks_per_sm);

  // TODO: make user pass in working memory? This allows integration
  //       with CNMeM (used by Theano)
  int reduction_mem_sz = 2 * n_blocks * 33 * n_dims * sizeof(float);
  float *d_reduction_mem;
  gpuErrChk(cudaMalloc(&d_reduction_mem, reduction_mem_sz));
  float *d_decay_storage = &d_reduction_mem[0 * n_blocks * 33 * n_dims];
  float *d_h_storage = &d_reduction_mem[1 * n_blocks * 33 * n_dims];

  // TODO: run kernels on non-default stream?
  reduction_kernel_fast<<<n_blocks, 1024>>>(decays, impulses, initial_state,
               d_decay_storage, d_h_storage,
               n_dims, n_steps);

  int n_scan_blocks = (n_dims + SCAN_ARRS_PER_BLK - 1) / SCAN_ARRS_PER_BLK;
  block_scan_kernel_fast<<<n_scan_blocks, SCAN_BLOCK_DIM>>>(d_decay_storage, d_h_storage,
          n_dims, n_blocks);

  warp_scan_kernel_fast<<<n_blocks, 1024>>>(decays, impulses,
               initial_state, out,
               d_decay_storage, d_h_storage,
               n_dims, n_steps);

  gpuErrChk(cudaFree(d_reduction_mem));
}
}

void test() {
  int n_dims = 100;
  int n_steps = 1000000;
  int n_elements = n_dims * n_steps;

  float *decays = (float *) calloc(n_elements, sizeof(float));
  for (int i = 0; i < n_elements; i++) {
    decays[i] = .999;
  }
  float *d_decays;
  gpuErrChk(cudaMalloc(&d_decays, n_elements * sizeof(float)));
  gpuErrChk(cudaMemcpy(d_decays, decays, n_elements * sizeof(float),
           cudaMemcpyHostToDevice));

  float *impulses = (float *) calloc(n_elements, sizeof(float));
  for (int i = 0; i < n_dims; i++) {
    impulses[i + 0 * n_dims] = 2.0;
  }
  float *d_impulses;
  gpuErrChk(cudaMalloc(&d_impulses, n_elements * sizeof(float)));
  gpuErrChk(cudaMemcpy(d_impulses, impulses,
           n_elements * sizeof(float), cudaMemcpyHostToDevice));

  float *out = (float *) calloc(n_elements, sizeof(float));
  float *d_out;
  gpuErrChk(cudaMalloc(&d_out, n_elements * sizeof(float)));
  gpuErrChk(cudaMemset(d_out, 0, n_elements * sizeof(float)));

  compute_fast_linear_recurrence(d_decays, d_impulses, NULL, d_out, n_dims, n_steps);
  gpuErrChk(cudaMemcpy(out, d_out, n_elements * sizeof(float),
           cudaMemcpyDeviceToHost));

  gpuErrChk(cudaFree(d_decays));
  gpuErrChk(cudaFree(d_impulses));
  gpuErrChk(cudaFree(d_out));
}
