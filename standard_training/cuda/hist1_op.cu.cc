#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void Hist11D(const float* grad, const float* input, const float* softass,
		        const int nbins, const int nbatch, const int ndepth, const float alpha,
  	                float* output){
  CUDA_1D_KERNEL_LOOP(idx, ndepth) {
    for (int i = idx; i < nbatch * ndepth; i += ndepth) {
      output[i] = 0;
    }

    for (int i = idx; i < nbatch * ndepth; i += ndepth) {
      int cur_idx = round(input[i]);
      cur_idx = cur_idx < 0 ? 0 : cur_idx;
      cur_idx = cur_idx > nbins - 1 ? nbins - 1 : cur_idx;

      float soft_ass1 = softass[i];
      float soft_ass2 = softass[i + nbatch * ndepth];
      
      // the gradient computation w.r.t softmax 
      float dy1x = soft_ass1 * (1 - soft_ass1) * 2 * alpha * (input[i] - cur_idx)
                 - soft_ass1 * soft_ass2 * 2 * alpha * (input[i] - (1 + cur_idx));

      float dy2x = soft_ass2 * (1 - soft_ass2) * 2 * alpha * (input[i] - (1 + cur_idx))
                 - soft_ass1 * soft_ass2 * 2 * alpha * (input[i] - cur_idx);

      // multiply the computed gradient with the backprop from the above layer
      output[i] += grad[cur_idx * ndepth + idx] * dy1x  
                + grad[(cur_idx + 1) * ndepth + idx] * dy2x;   
    }
  }	   
}


void Hist11DLauncher(const float* grad, const float* input, const float* softass,
		     const int nbins, const int nbatch, const int ndepth, const float alpha,
  	             float* output) {
  
  const int blockSize = 256;
  const int numBlocks = (ndepth + blockSize - 1) / blockSize;
  Hist11D<<<numBlocks, blockSize>>>(grad, input, softass, 
				    nbins, nbatch, ndepth, alpha,
			            output);
}


#endif



