#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <math.h>
#include <cfloat>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void Indices1D(const float* source, const float* scumul, const float* tcumul,
			  const int nbatch, const int ndepth, const int nbins,       
			  float* nn_idx1, int* idx11, int* idx12,
			  float* nn_idx2, int* idx21, int* idx22){

  CUDA_1D_KERNEL_LOOP(index, ndepth) {   
    
    for (int i = index; i < (nbins + 1) * ndepth; i += ndepth) { 
      int NN_Idx = 0;                  
      float dist = abs(scumul[i] - tcumul[index]);
      for ( int n = 1; n < nbins + 1; ++n ) {
        float newDist = abs(scumul[i] - tcumul[n * ndepth + index]);
        if ( newDist > dist ) {
	  break;
        } else {
            dist = newDist;
            NN_Idx = n;
        }
      } 
      nn_idx1[i] = NN_Idx > nbins - 1? nbins - 1: NN_Idx;
      idx11[i] = nn_idx1[i] * ndepth + index;
      idx12[i] = (nn_idx1[i] + 1) * ndepth + index;
    }

    //search nearest neighbor index          
    for (int i = index; i < nbatch * ndepth; i += ndepth) {
      nn_idx2[i] = round(source[i]) > nbins - 1? nbins - 1: round(source[i]);	
      idx21[i] = nn_idx2[i] * ndepth + index;
      idx22[i] = (nn_idx2[i] + 1) * ndepth + index;
    }
  }	   
}

void Indices1DLauncher(const float* source, const float* scumul, const float* tcumul,
		       const int nbatch, const int ndepth, const int nbins,       
		       float* nn_idx1, int* idx11, int* idx12,
		       float* nn_idx2, int* idx21, int* idx22) {
  
  const int blockSize = 256;
  const int numBlocks = (ndepth + blockSize - 1) / blockSize;
  Indices1D<<<numBlocks, blockSize>>>(source, scumul, tcumul,
				      nbatch, ndepth, nbins,
				      nn_idx1, idx11, idx12,
				      nn_idx2, idx21, idx22);
}
#endif



