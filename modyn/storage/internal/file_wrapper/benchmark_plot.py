import re
import matplotlib.pyplot as plt

python_times = []
cpp_times = []
num_samples = []

with open("output.txt", "r") as f:
    for line in f.readlines():
        if "Python microbenchmark time" in line:
            python_time = float(line.split(':')[-1].strip().split()[-4])
            python_times.append(float(python_time))
        elif "C++ microbenchmark time" in line:
            cpp_time = float(line.split(':')[-1].strip().split()[-4])
            cpp_times.append(float(cpp_time))
        elif "Running microbenchmark" in line:
            num_sample = float(line.split(':')[-1].strip().split()[-3])
            num_samples.append(float(num_sample))

plt.plot(num_samples, python_times, label="Python")
plt.plot(num_samples, cpp_times, label="C++")
plt.xlabel("Number of samples")
plt.ylabel("Time (ms)")
plt.legend()
plt.show()