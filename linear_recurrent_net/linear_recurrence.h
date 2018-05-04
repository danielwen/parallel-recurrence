extern "C" {
void compute_fast_linear_recurrence(
  const float *decays,          /* n_steps x n_dims row major matrix */
  const float *impulses,        /* n_steps x n_dims row major matrix */
  const float *initial_state,   /* n_dims vector */
  float *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                   /* dimensionality of recurrent vector */
  int n_steps                   /* length of input and output sequences */
);

void compute_linear_recurrence_baseline(
  const float *decays,          /* n_steps x n_dims row major matrix */
  const float *impulses,        /* n_steps x n_dims row major matrix */
  const float *initial_state,   /* n_dims vector */
  float *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                   /* dimensionality of recurrent vector */
  int n_steps                   /* length of input and output sequences */
);

void compute_serial_linear_recurrence_baseline(
  const float *decays,          /* n_steps x n_dims row major matrix */
  const float *impulses,        /* n_steps x n_dims row major matrix */
  const float *initial_state,   /* n_dims vector */
  float *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                   /* dimensionality of recurrent vector */
  int n_steps                   /* length of input and output sequences */
);
}

void test_fast();
void test_base();

