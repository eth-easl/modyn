import json

import matplotlib.pyplot as plt

# Load the statistics from the JSON file
with open("hierarchy_stats.json", "r") as f:
    stats = json.load(f)

split = "train"


# Get the list of all classes
all_classes = stats[split]["per_class"].keys()
num_classes = len(set(all_classes))

print(f"there are {len(set(all_classes))} classes")

# Get the years available in the dataset
years = sorted(stats[split]["per_year_and_class"].keys())

num_years = len(years)

results = [0 for _ in range(num_years + 1)]

# Plot the number of samples per year for each class
for class_name in [str(i) for i in range(num_classes)]:
    samples_per_year = [stats[split]["per_year_and_class"].get(year, {}).get(class_name, 0) for year in years]

    max_samples = -1
    max_year_idx = -1
    for year_idx, samples in enumerate(samples_per_year):
        if samples > max_samples:
            max_year_idx = year_idx
            max_samples = samples

    for i in range(len(results)):
        base_idx = max_year_idx + i - (num_years // 2)
        if base_idx < 0 or base_idx >= len(samples_per_year):
            continue
        results[i] += samples_per_year[base_idx] / sum(samples_per_year)

print(results)
plt.figure(figsize=(10, 5))
plt.bar(range(-num_years // 2, (num_years // 2) + 1, 1), results, color="cornflowerblue")
plt.title(f"Number of Samples per Year for Class: {class_name}")
plt.xlabel("Year")
plt.ylabel("Number of Samples")
plt.xticks(range(-num_years // 2, (num_years // 2) + 1, 1), rotation=45)
plt.tight_layout()
plt.show()
