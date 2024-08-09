import json

import matplotlib.pyplot as plt
import numpy as np

# Load the statistics from the JSON file
with open("dataset_stats.json") as f:
    stats = json.load(f)

# Sort classes by sample count and get the top 20
class_counts = stats["train"]["per_class"]

print("total classes: " + str(len(class_counts)))

sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)


print("total sorted classes: " + str(len(class_counts)))

top_classes = sorted_classes[:50]
top_class_names, top_class_samples = zip(*top_classes)
num_top_samples = sum(count for _, count in top_classes)

print(f"total top classes: {len(top_classes)} num_samples in there: {num_top_samples}")

# Sum the samples of all other classes
other_samples = sum(count for _, count in sorted_classes[50:])
avg_samples_class = other_samples / len(sorted_classes[50:])
print("average samples / class in other: " + str(avg_samples_class))
print("remaining classes: " + str(len(sorted_classes[50:])))

# Print the top classes and their sample counts for verification
print("Top classes and their sample counts:")
for class_name, sample_count in top_classes:
    print(f"{class_name}: {sample_count}")
print(f"Other: {other_samples}")

# Plot bar chart of samples per class for top 20 classes and "Other"
plt.figure(figsize=(14, 6))
plt.bar(top_class_names + ("Other",), top_class_samples + (other_samples,), color="skyblue")
plt.title("Number of Samples per Top 20 Classes and Other")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Prepare data for stacked bar chart
years = sorted(stats["train"]["per_year_and_class"].keys())
top_classes_set = set(top_class_names)

# Initialize the data structure for stacked bar chart
stacked_data = {year: {class_name: 0 for class_name in top_class_names + ("Other",)} for year in years}

# Populate the data structure with actual counts
for year, classes_in_year in stats["train"]["per_year_and_class"].items():
    for class_name, count in classes_in_year.items():
        if class_name in top_classes_set:
            stacked_data[year][class_name] += count
        else:
            stacked_data[year]["Other"] += count

# Plot stacked bar chart of samples per class within each year for top 20 classes and "Other"
plt.figure(figsize=(14, 6))
bottom = np.zeros(len(years))

for class_name in top_class_names + ("Other",):
    samples_per_year = [stacked_data[year][class_name] for year in years]
    plt.bar(years, samples_per_year, bottom=bottom, label=class_name)
    bottom = np.add(bottom, samples_per_year)

plt.title("Number of Samples per Top 20 Classes and Other within Each Year")
plt.xlabel("Year")
plt.ylabel("Number of Samples")
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
