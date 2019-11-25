#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <functional>

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TmciCheckpoint")
    .Attr("backend: string")
    .Attr("config: string")
    .Input("tensors: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            return Status::OK();
    });

class TMCICheckpointOp : public OpKernel {

    public:

    explicit TMCICheckpointOp(OpKernelConstruction* context)
    : OpKernel(context) {
        std::string backend;
        std::string config;
        OP_REQUIRES_OK(context, context->GetAttr("backend", &backend));
        OP_REQUIRES_OK(context, context->GetAttr("config", &config));
    }

    void Compute(OpKernelContext* context) override {
        unsigned n = context->num_inputs();
        std::vector<std::pair<void*, size_t>> segments(n);
        for(unsigned i=0; i < n; i++) {
            auto& tensor = context->input(i);
            segments[i].first  = static_cast<void*>(
                    const_cast<char*>(tensor.tensor_data().data()));
            segments[i].second = tensor.tensor_data().size();
        }
/*
        OP_REQUIRES(context, status.first == 0,
                errors::Internal(status.second));
                */
    }

};

REGISTER_KERNEL_BUILDER(Name("TmciCheckpoint").Device(DEVICE_CPU), TMCICheckpointOp);

