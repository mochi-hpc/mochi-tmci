#ifndef __DUMMY_BACKEND_HPP
#define __DUMMY_BACKEND_HPP

#include <tmci/backend.hpp>

class DummyBackend : public tmci::Backend {

    public:

    DummyBackend(const char* config) {
        std::cout << "[DUMMY] Initialization, config: " << config << std::endl;
    }

    ~DummyBackend() = default;

    virtual int Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
        std::cout << "[DUMMY] Saving " << tensors.size() << " tensors:" << std::endl;
        for(const tensorflow::Tensor& t : tensors) {
            std::cout << "  data=" << (void*)t.tensor_data().data() << " size=" << t.tensor_data().size() << std::endl;
        }
        return 0;
    }
    virtual int Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) {
        std::cout << "[DUMMY] Loading " << tensors.size() << " tensors:" << std::endl;
        for(const tensorflow::Tensor& t : tensors) {
            std::cout << "  data=" << (void*)t.tensor_data().data() << " size=" << t.tensor_data().size() << std::endl;
        }
        return 0;
    }
};

#endif
