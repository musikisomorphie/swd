#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void Hist1D(const float* input,
		       const int nbins, const int nbatch, const int ndepth, const float alpha,
  	               float* hist, float* output){
  CUDA_1D_KERNEL_LOOP(idx, ndepth) {
    for (int i = idx; i < (nbins + 1) * ndepth; i += ndepth) {
      hist[i] = 0;
    } 

    for (int i = idx; i < nbatch * ndepth; i += ndepth) {
      // find which bin the input should assign
      int cur_idx = round(input[i]);
      cur_idx = cur_idx < 0 ? 0 : cur_idx;
      cur_idx = cur_idx > nbins - 1 ? nbins - 1 : cur_idx;
      
      // soft assign for two neighboring bins
      float soft_ass1 = exp(alpha * (input[i] - cur_idx) * (input[i] - cur_idx));
      float soft_ass2 = exp(alpha * (input[i] - (1 + cur_idx)) * (input[i] - (1 + cur_idx)));

      // intermediate result for gradient computation(hist1) 
      output[i] = soft_ass1 / (soft_ass1 + soft_ass2);
      output[i + nbatch * ndepth] = soft_ass2 / (soft_ass1 + soft_ass2);

      // compute histogram	
      hist[cur_idx * ndepth + idx] += output[i];
      hist[(cur_idx + 1) * ndepth + idx] += output[i + nbatch * ndepth];
    }
  }	   
}


void Hist1DLauncher(const float* input,
		    const int nbins, const int nbatch, const int ndepth, const float alpha,
  	            float* hist, float* output) {
  
  const int blockSize = 256;
  const int numBlocks = (ndepth + blockSize - 1) / blockSize;
  Hist1D<<<numBlocks, blockSize>>>(input,
				   nbins, nbatch, ndepth, alpha,
			           hist, output);
}


#endif



