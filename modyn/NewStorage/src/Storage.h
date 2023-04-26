#include <yaml-cpp/yaml.h>
#include <string>

namespace storage {
    class Storage {
        private: 
            YAML::Node config;
        public:
            Storage(std::string config_file);
            void run();
    };
}