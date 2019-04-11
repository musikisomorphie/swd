#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void Gat11D(const float* grad, const int* idx,
		       const int* in_sz, const int* ou_sz, const int ndepth, 
		       float* output){

  CUDA_1D_KERNEL_LOOP(index, ndepth) {
    for (int i = index; i < in_sz[0] * ndepth; i += ndepth) {
      output[i] = 0;
    } 
      
    for (int i = index; i < ou_sz[0] * ndepth; i += ndepth) { 
      int cur_idx = idx[i];
      output[cur_idx] += grad[i];  
    }
  }	   
}


void Gat11DLauncher(const float* grad, const int* idx,
		    const int* in_sz, const int* ou_sz, const int ndepth, 
		    float* output) {
  
  const int blockSize = 256;
  const int numBlocks = (ndepth + blockSize - 1) / blockSize;
  Gat11D<<<numBlocks, blockSize>>>(grad, idx,
				   in_sz, ou_sz, ndepth, 
			           output);
}


#endif



