#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <functional>
#include "backend.hpp"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TmciRestore")
    .Attr("backend: string")
    .Attr("config: string")
    .Input("tensors: Ref(N * T)")
    .Attr("T: type")
    .Attr("N: int")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            return Status::OK();
    });

class TMCIRestoreOp : public OpKernel {

    public:

    explicit TMCIRestoreOp(OpKernelConstruction* context)
    : OpKernel(context) {
        std::string backend;
        std::string config;
        OP_REQUIRES_OK(context, context->GetAttr("backend", &backend));
        OP_REQUIRES_OK(context, context->GetAttr("config", &config));
        for(int i=0; i < context->num_inputs(); i++) {
            OP_REQUIRES(context, IsRefType(context->input_type(i)),
                errors::InvalidArgument("input tensor needs to be a ref type"));
        }
        m_backend = tmci::Backend::Create(backend.c_str(), config.c_str());
    }

    void Compute(OpKernelContext* context) override {
        unsigned n = context->num_inputs();
        std::vector<tensorflow::Tensor> tensors;
        tensors.reserve(n);
        for(unsigned i=0; i < n; i++) {
            tensors.push_back(context->mutable_input(i, false));
        }
        int status = m_backend->Load(tensors);
        OP_REQUIRES(context, status == 0, errors::Internal(status));
    }

    private:

    std::unique_ptr<tmci::Backend> m_backend;

};

REGISTER_KERNEL_BUILDER(Name("TmciRestore").Device(DEVICE_CPU), TMCIRestoreOp);
