#include <fstream>

void create_dummy_yaml() {
    std::ofstream out("config.yaml");
    out << "test: 1" << std::endl;
    out.close();
}

void delete_dummy_yaml() {
    std::remove("config.yaml");
}