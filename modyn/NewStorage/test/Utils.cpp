#include "Utils.hpp"

using namespace storage;

void Utils::create_dummy_yaml()
{
    std::ofstream out("config.yaml");
    out << "test: 1" << std::endl;
    out.close();
}
void Utils::delete_dummy_yaml()
{
    std::remove("config.yaml");
}
YAML::Node Utils::get_dummy_config()
{
    YAML::Node config;
    config["file_extension"] = ".txt";
    config["label_file_extension"] = ".json";
    config["label_size"] = 1;
    config["sample_size"] = 2;
    return config;
}
