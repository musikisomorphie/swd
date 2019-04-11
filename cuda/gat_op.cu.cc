#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void Gat1D(const float* input, const int* idx,
		      const int* ou_sz, const int ndepth, float* output){

  CUDA_1D_KERNEL_LOOP(index, ndepth) {       
    for (int i = index; i < ou_sz[0] * ndepth; i += ndepth) { 
      int cur_idx = idx[i];
      output[i] = input[cur_idx];  
    }
  }	   
}

void Gat1DLauncher(const float* input, const int* idx,
		   const int* ou_sz, const int ndepth, float* output) {
  
  const int blockSize = 256;
  const int numBlocks = (ndepth + blockSize - 1) / blockSize;
  Gat1D<<<numBlocks, blockSize>>>(input, idx,
				  ou_sz, ndepth, output);
}

#endif



