#! /bin/sh
rm -rf lib/

mkdir lib
nvcc -c linear_recurrence_base.cu -o lib/linear_recurrence_base.o -O3 --compiler-options '-fPIC' # --device-c
nvcc -c linear_recurrence_fast.cu -o lib/linear_recurrence_fast.o -O3 --compiler-options '-fPIC' # --device-c
nvcc lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -shared -o lib/liblinear_recurrence.so --compiler-options '-fPIC' # -rdc=true

# building tensorflow op
export TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
export CUDA_HOME=/usr/local/cuda
g++ -std=c++11 -shared -o lib/tf_linear_recurrence.so tensorflow_binding/linear_recurrence_op.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L $CUDA_HOME/lib64 -O3 -I $TF_INC -I $TF_INC/external/nsync/public -fPIC -lcudart -L $TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
