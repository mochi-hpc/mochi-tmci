#include "backend.hpp"
#include <unordered_map>

namespace tmci {

std::unordered_map<std::string, std::function<std::unique_ptr<Backend>(const char*)>> _backend_factories;

Backend::Backend() {}

Backend::~Backend() {}

std::unique_ptr<tmci::Backend> Backend::Create(const char* name, const char* config) {
    if(_backend_factories.count(name) == 0)
        throw std::invalid_argument(std::string("TMCI backend \"") + name + "\" not found");
    return _backend_factories[name](config);
}

void Backend::RegisterFactory(const char* name, 
            std::function<std::unique_ptr<Backend>(const char*)>&& factory) {
    if(_backend_factories.count(name) != 0)
        throw std::runtime_error(std::string("TMCI backend \"") + name + "\" already registered");
    _backend_factories[name] = std::move(factory);
}

}
