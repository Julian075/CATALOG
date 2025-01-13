import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data from a file
with open('Ablation_omg.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a plottable format
x = list(map(float, data.keys()))  # Convert the keys to float for the x-axis
y_values = list([[data[key][1], data[key][3]] for key in data])  # Extract cis_test_acc and trans_test_acc

# Plot cis_test_acc and trans_test_acc
plt.figure(figsize=(10, 6))
for i, label in enumerate(["cis_test_acc", "trans_test_acc"]):
    y = [vals[i] if vals[i] is not None else np.nan for vals in y_values]
    plt.plot(x, y, marker="o", label=label)

# Add labels, legend, and title
plt.xlabel("Beta Values")
plt.ylabel("Accuracy")
plt.title("Templates (Beta) + Description_LLM (Beta)")
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("Ablation_omg.png")
plt.show()