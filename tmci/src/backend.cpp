#include "backend.hpp"
#if 0
#include <pybind11/pybind11.h>
#endif
#include <unordered_map>

namespace tmci {

std::unordered_map<std::string, std::function<std::unique_ptr<Backend>(const char*)>> _backend_factories;

Backend::Backend() {}

Backend::~Backend() {}

std::unique_ptr<tmci::Backend> Backend::Create(const char* name, const char* config) {
    if(_backend_factories.count(name) == 0)
        throw std::runtime_error("TMCI backend not found");
    return _backend_factories[name](config);
}

void Backend::RegisterFactory(const char* name, 
            std::function<std::unique_ptr<Backend>(const char*)>&& factory) {
    if(_backend_factories.count(name) != 0)
        throw std::runtime_error("TMCI backend already registered");
    _backend_factories[name] = std::move(factory);
}

}
