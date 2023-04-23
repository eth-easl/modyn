# Run and time the binary file wrapper microbenchmark.

# The microbenchmark is a simple program that creates a file, writes binary data to it, reads the data using the python and C++ wrappers, and then deletes the file. The program is run 100 times and the average time is reported.

# Create a temporary directory to store the test file.
mkdir -p /tmp/modyn_test
cd /tmp/modyn_test

# Create a file with random data of the form [LABEL, DATA, LABEL, DATA, ...] where LABEL is a 4-byte integer and DATA is a 4-byte integer. The byte order is big-endian.
function create_random_file {
    python3 -c "import random; import struct; encoded_integers = b''.join(struct.pack('<I', random.randint(0, 2147483647)) for _ in range(2*int($1))); padding = b'\x00' * ((2 * int($1) * 4) - len(encoded_integers)); encoded_data = encoded_integers + padding; open('data.bin', 'wb').write(encoded_data)"
}

function run_python_microbenchmark {
    echo "Running python microbenchmark"

    # Run the microbenchmark 100 times and report the average time
    local time=$(python3 -m timeit -r 1 -u msec -n 10 -s "import modyn.storage.internal.file_wrapper.binary_file_wrapper_microbenchmark as microbenchmark" "microbenchmark.run()")
    echo "Python microbenchmark time: $time"
}

function run_cpp_microbenchmark() {
    echo "Running C++ microbenchmark"

    g++ -std=c++17 -O3 -o /Users/viktorgsteiger/Documents/modyn/modyn/storage/internal/file_wrapper/binary_file_wrapper /Users/viktorgsteiger/Documents/modyn/modyn/storage/internal/file_wrapper/binary_file_wrapper.cpp
    /Users/viktorgsteiger/Documents/modyn/modyn/storage/internal/file_wrapper/binary_file_wrapper 10 "${1}" data.bin 
    rm /Users/viktorgsteiger/Documents/modyn/modyn/storage/internal/file_wrapper/binary_file_wrapper
}

function benchmark {
    for i in `seq 1000000 500000 10000000`; do
        echo 'Running microbenchmark with ' $i ' label-data pairs'
        create_random_file "${i}"
        run_python_microbenchmark
        run_cpp_microbenchmark "${i}"
        rm data.bin
    done
}

benchmark > /Users/viktorgsteiger/Documents/modyn/modyn/storage/internal/file_wrapper/microbenchmark_results.txt

# Clean up
rm -rf /tmp/modyn_test
