import csv
import matplotlib.pyplot as plt

def load_log(path):
    steps = []
    upright = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            upright.append(int(row["upright"]))
    return steps, upright


files = {
    "Baseline": "results/logs/double_v1_baseline_eval.csv",
    "Reward Shaping": "results/logs/double_v2_reward_shaping_eval.csv"
}

plt.figure(figsize=(10, 4))

for label, path in files.items():
    steps, upright = load_log(path)
    # moving average for clarity
    window = 50
    smooth = [
        sum(upright[i:i+window]) / window
        for i in range(len(upright) - window)
    ]
    plt.plot(smooth, label=label)

plt.xlabel("Time steps")
plt.ylabel("Upright probability (moving average)")
plt.title("Double Inverted Pendulum Evaluation Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/plots/double_pendulum_comparison.png")
plt.show()