#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include <stdio.h>
//#include <cfloat>


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Indices")
    .Input("source: float")
    .Input("scumul: float")
    .Input("tcumul: float")
    .Attr("nbatch: int")
    .Attr("ndepth: int")
    .Attr("nbins: int")
    .Output("nn_idx1: float")
    .Output("idx11: int32")
    .Output("idx12: int32")
    .Output("nn_idx2: float")
    .Output("idx21: int32")
    .Output("idx22: int32");
    //.SetShapeFn([](shape_inference::InferenceContext* c) {
    //  c->set_output(0, c->input(0));
    //  return Status::OK();
    //});


void Indices1DLauncher(const float* source, const float* scumul, const float* tcumul,
		       const int nbatch, const int ndepth, const int nbins,       
		       float* nn_idx1, int* idx11, int* idx12,
		       float* nn_idx2, int* idx21, int* idx22); 


class IndicesOp : public OpKernel {
 public:
  explicit IndicesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("nbatch", &nbatch_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("ndepth", &ndepth_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("nbins", &nbins_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& source_tensor = context->input(0);
    const Tensor& scumul_tensor = context->input(1);
    const Tensor& tcumul_tensor = context->input(2);    
    auto source = source_tensor.flat<float>();
    auto scumul = scumul_tensor.flat<float>();
    auto tcumul = tcumul_tensor.flat<float>();
    
    // Initialize the output 
    Tensor* nn_idx1_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({(nbins_ + 1), ndepth_}),
                                                     &nn_idx1_tensor));
    auto nn_idx1 = nn_idx1_tensor->template flat<float>();

    Tensor* idx11_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({(nbins_ + 1), ndepth_}),
                                                     &idx11_tensor));
    auto idx11 = idx11_tensor->template flat<int>();
    
    Tensor* idx12_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({(nbins_ + 1), ndepth_}),
                                                     &idx12_tensor));
    auto idx12 = idx12_tensor->template flat<int>();

    Tensor* nn_idx2_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({nbatch_, ndepth_}),
                                                     &nn_idx2_tensor));
    auto nn_idx2 = nn_idx2_tensor->template flat<float>();

    Tensor* idx21_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({nbatch_, ndepth_}),
                                                     &idx21_tensor));
    auto idx21 = idx21_tensor->template flat<int>();
    
    Tensor* idx22_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape({nbatch_, ndepth_}), 
						     &idx22_tensor));
    auto idx22 = idx22_tensor->template flat<int>();
		    
    Indices1DLauncher(source.data(), scumul.data(), tcumul.data(),
		      nbatch_, ndepth_, nbins_,			   
		      nn_idx1.data(), idx11.data(), idx12.data(),
		      nn_idx2.data(), idx21.data(), idx22.data());
    
  }
  private:
    int nbatch_, ndepth_, nbins_; 
};

REGISTER_KERNEL_BUILDER(Name("Indices").Device(DEVICE_GPU), IndicesOp);
