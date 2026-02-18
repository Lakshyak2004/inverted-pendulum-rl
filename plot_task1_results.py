import csv
import matplotlib.pyplot as plt

LOG_FILE = "results/logs/task1_eval.csv"
PLOT_FILE = "results/plots/task1_upright_probability.png"

steps = []
upright = []

with open(LOG_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        upright.append(int(row["upright"]))

# Moving average for smooth plot
window = 50
upright_avg = [
    sum(upright[i:i+window]) / window
    for i in range(len(upright) - window)
]

plt.figure(figsize=(8, 4))
plt.plot(upright_avg)
plt.xlabel("Time step")
plt.ylabel("Upright probability (moving average)")
plt.title("Task 1: Swing-up and Stabilization Performance")
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_FILE)
plt.show()

print("Saved plot to:", PLOT_FILE)
