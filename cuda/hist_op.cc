#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include <stdio.h>
//#include <cfloat>


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Hist")
    .Input("input: float")
    .Attr("nbins: int")
    .Attr("nbatch: int")
    .Attr("ndepth: int")
    .Attr("alpha: float")
    .Output("hist: float")
    .Output("output: float");
    //.SetShapeFn([](shape_inference::InferenceContext* c) {
    //  c->set_output(0, c->input(0));
    //  return Status::OK();
    //});


void Hist1DLauncher(const float* input,
		    const int nbins, const int nbatch, const int ndepth, const float alpha,
  	            float* hist, float* output); 


class HistOp : public OpKernel {
 public:
  explicit HistOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("nbins", &nbins_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("nbatch", &nbatch_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("ndepth", &ndepth_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("alpha", &alpha_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    
    // Initialize the output 
    Tensor* hist_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({(nbins_ + 1), ndepth_}),
                                                     &hist_tensor));
    auto hist = hist_tensor->template flat<float>();

    Tensor* output_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({2, nbatch_, ndepth_}),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();
    
    Hist1DLauncher(input.data(),
	           nbins_, nbatch_, ndepth_, alpha_,
	  	   hist.data(), output.data());
    
  }
  private:
    int nbins_, nbatch_, ndepth_; 
    float alpha_;
};

REGISTER_KERNEL_BUILDER(Name("Hist").Device(DEVICE_GPU), HistOp);
