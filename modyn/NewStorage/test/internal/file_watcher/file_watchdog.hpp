#include <yaml-cpp/yaml.h>
#include <map>
#include <tuple>
#include <boost/process.hpp>

namespace storage
{
    class FileWatchdog
    {
    private:
        YAML::Node config;
        std::map<int, std::tuple<std::boost::child, int>> file_watcher_processes;
        void watch_file_watcher_processes();
        void start_file_watcher_process(int dataset_id);
        void stop_file_watcher_process(int dataset_id);

    public:
        FileWatchdog(YAML::Node config)
        {
            this->config = config;
            this->file_watcher_processes = {};
        }
        void run();
    };
}