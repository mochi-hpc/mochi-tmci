#ifndef __TMCI_BACKEND_HPP
#define __TMCI_BACKEND_HPP

#include <vector>
#include <memory>
#include <functional>
#include <tensorflow/core/framework/tensor.h>

namespace tmci {

template<typename T>
class BackendRegistry;

class Backend {

    public:

    Backend();

    virtual ~Backend();

    virtual int Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors) = 0;

    virtual int Load(const std::vector<tensorflow::Tensor>& tensors) = 0;

    static std::unique_ptr<Backend> Create(const char* name, const char* config);

    private:

    template<typename T>
    friend class BackendRegistry;

    static void RegisterFactory(const char* name,
            std::function<std::unique_ptr<Backend>(const char*)>&& factory);
};

template<typename T>
class BackendRegistry {

    public:

    BackendRegistry(const char* backend_name) {
        Backend::RegisterFactory(backend_name, [](const char* config) {
            return std::make_unique<T>(config);
        });
    }
};

}

#define TMCI_REGISTER_BACKEND(__name, __class) \
    static tmci::BackendRegistry<__class> __tmci_registration(__name);

#endif
