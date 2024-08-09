import json

import matplotlib.pyplot as plt

# Load the statistics from the JSON file
with open("hierarchy_stats.json") as f:
    stats = json.load(f)

split = "train"
num_classes = 79


# Get the list of all classes
all_classes = stats[split]["per_class"].keys()

print(f"there are {len(set(all_classes))} classes")

# Get the years available in the dataset
years = sorted(stats[split]["per_year_and_class"].keys())

# Plot the number of samples per year for each class
for class_name in [str(i) for i in range(num_classes)]:
    samples_per_year = [stats[split]["per_year_and_class"].get(year, {}).get(class_name, 0) for year in years]

    plt.figure(figsize=(10, 5))
    plt.bar(years, samples_per_year, color="cornflowerblue")
    plt.title(f"Number of Samples per Year for Class: {class_name}")
    plt.xlabel("Year")
    plt.ylabel("Number of Samples")
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()
