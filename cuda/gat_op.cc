#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include <stdio.h>
//#include <cfloat>


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Gat")
    .Input("input: float")
    .Input("idx: int32")
    .Input("in_sz: int32")
    .Input("ou_sz: int32")
    .Attr("ndepth: int")
    .Output("output: float");
    //.SetShapeFn([](shape_inference::InferenceContext* c) {
    //  c->set_output(0, c->input(0));
    //  return Status::OK();
    //});

void Gat1DLauncher(const float* input, const int* idx,
		   const int* ou_sz, const int ndepth, float* output); 


class GatOp : public OpKernel {
 public:
  explicit GatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("ndepth", &ndepth_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& idx_tensor = context->input(1);
    auto input = input_tensor.flat<float>();
    auto idx = idx_tensor.flat<int>();

    const Tensor& in_sz_tensor = context->input(2);
    const Tensor& ou_sz_tensor = context->input(3);
    auto in_sz = in_sz_tensor.flat<int>();
    auto ou_sz = ou_sz_tensor.flat<int>();
    
    // Initialize the output 
    Tensor* output_tensor = NULL; 
    OP_REQUIRES_OK(context, context->allocate_output(0, idx_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();
		    
    Gat1DLauncher(input.data(), idx.data(), 
		  ou_sz.data(), ndepth_, output.data());
    
  }
  private:
    int ndepth_; 
};

REGISTER_KERNEL_BUILDER(Name("Gat").Device(DEVICE_GPU), GatOp);


