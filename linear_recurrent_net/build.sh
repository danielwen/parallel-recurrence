#!/bin/bash
rm -f correct
rm -f profile
rm -rf lib/

DEBUG=0
TEST=0

# ./build.sh DEBUG TEST 
for arg in "$@"; do
	if [[ "$arg" == "DEBUG" ]]; then 
		echo "Using DEBUG mode";
		DEBUG=1; 
	fi
	if [[ "$arg" == "TEST" ]]; then 
		echo "Building TEST mode with DEBUG"
		TEST=1; 
		DEBUG=1;
	fi
done 

mkdir lib
nvcc -c linear_recurrence_base.cu -o lib/linear_recurrence_base.o -DDEBUG=$DEBUG -O3 --compiler-options '-fPIC' # --device-c
nvcc -c linear_recurrence_fast.cu -o lib/linear_recurrence_fast.o -DDEBUG=$DEBUG -O3 --compiler-options '-fPIC' # --device-c
nvcc lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -shared -o lib/liblinear_recurrence.so --compiler-options '-fPIC' # -rdc=true

# building tensorflow op
export TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
export CUDA_HOME=/usr/local/cuda
export MISC_DIR=/usr/local/cuda-8.0/targets/x86_64-linux/lib/ # For a different machine, can ignore
g++ -std=c++11 -shared -o lib/tf_linear_recurrence.so tensorflow_binding/linear_recurrence_op.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L $CUDA_HOME/lib64 -O3 -I $TF_INC -I $TF_INC/external/nsync/public -fPIC -L $MISC_DIR -lcudart -L $TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

# testing code

g++ -o correct correct.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L/usr/local/cuda/lib64 -L $CUDA_HOME/lib64 -L $MISC_DIR -lcuda -lcudart 
g++ -o profile profile.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L/usr/local/cuda/lib64 -L $CUDA_HOME/lib64 -L $MISC_DIR -lcuda -lcudart 
# if [[ "$TEST" == "1" ]]; then
# 	echo -e "\nMaking correct and profile"
# 	g++ -o correct correct.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L/usr/local/cuda/lib64 -L $CUDA_HOME/lib64 -L $MISC_DIR -lcuda -lcudart 
# 	g++ -o profile profile.cpp lib/linear_recurrence_base.o lib/linear_recurrence_fast.o -L/usr/local/cuda/lib64 -L $CUDA_HOME/lib64 -L $MISC_DIR -lcuda -lcudart 
# fi
