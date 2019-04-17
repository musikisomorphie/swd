#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include <stdio.h>
//#include <cfloat>


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Hist1")
    .Input("grad: float")
    .Input("input: float")
    .Input("softass: float")    
    .Attr("nbins: int")
    .Attr("nbatch: int")
    .Attr("ndepth: int")
    .Attr("alpha: float")
    .Output("output: float");
    //.SetShapeFn([](shape_inference::InferenceContext* c) {
    //  c->set_output(0, c->input(0));
    //  return Status::OK();
    //});


void Hist11DLauncher(const float* grad, const float* input, const float* softass,
		     const int nbins, const int nbatch, const int ndepth, const float alpha,
  	             float* output); 


class Hist1Op : public OpKernel {
 public:
  explicit Hist1Op(OpKernelConstruction* context) : OpKernel(context) {
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
    const Tensor& grad_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);
    const Tensor& softass_tensor = context->input(2);
    auto grad = grad_tensor.flat<float>();    
    auto input = input_tensor.flat<float>();
    auto softass = softass_tensor.flat<float>();
    
    // Initialize the output 
    Tensor* output_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nbatch_, ndepth_}),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();
    
    Hist11DLauncher(grad.data(), input.data(), softass.data(),
	            nbins_, nbatch_, ndepth_, alpha_,
	  	    output.data());
    
  }
  private:
    int nbins_, nbatch_, ndepth_; 
    float alpha_;
};

REGISTER_KERNEL_BUILDER(Name("Hist1").Device(DEVICE_GPU), Hist1Op);
