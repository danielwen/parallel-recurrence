#include <assert.h>
#include <stdio.h>

#define CEIL_DIV(x, y) ((x + y - 1) / y)
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);


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
  int n_blocks_per_sm = 2;

  // we want at least 32 elements per block, but no reason to run
  // with more than the maximum number of concurrent blocks
  int n_blocks = min(CEIL_DIV(n_steps, 32), n_SMs * n_blocks_per_sm);



  // TODO Write code


}

}

void test_fast() {
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
