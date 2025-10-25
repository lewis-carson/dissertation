import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Read the CSV file
csv_path = "/Users/lewis/diss/data/out/action_value-00000-of-02148_data.csv"
df = pd.read_csv(csv_path)

# Extract win_prob column and clean data
win_probs = pd.to_numeric(df['win_prob'], errors='coerce')

# Remove NaN values and extreme outliers
win_probs = win_probs.dropna()

# Filter out extremely large values that might be scientific notation parsing errors
# Keep only reasonable win probabilities (typically between -1 and 1 for normalized probs, or 0-1 for standard probs)
filtered_probs = win_probs[np.abs(win_probs) < 100]

print(f"Total data points: {len(df)}")
print(f"Valid numeric values: {len(win_probs)}")
print(f"Values within reasonable range: {len(filtered_probs)}")
print(f"Statistics for filtered win probabilities:")
print(f"  Min: {filtered_probs.min():.4f}")
print(f"  Max: {filtered_probs.max():.4f}")
print(f"  Mean: {filtered_probs.mean():.4f}")
print(f"  Std: {filtered_probs.std():.4f}")
print(f"  Median: {filtered_probs.median():.4f}")

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(filtered_probs, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Win Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Win Probabilities', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Save and show
output_path = "/Users/lewis/diss/data/eda/win_prob_histogram.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {output_path}")
plt.show()
