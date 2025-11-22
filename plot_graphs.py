"""
plot_graphs.py
Compare YOLOv8 CPU vs GPU training times and visualize performance difference
"""

import re
import os
import matplotlib.pyplot as plt

# === 1️⃣ Define paths ===
PROJECT_DIR = r"C:\Coding\GPU Project"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# Ensure logs folder exists
os.makedirs(LOG_DIR, exist_ok=True)

CPU_LOG = os.path.join(LOG_DIR, "train_cpu.log")
GPU_LOG = os.path.join(LOG_DIR, "train_gpu.log")

# === 2️⃣ Function to extract epoch times ===
def extract_epoch_times(log_path):
    """
    Extracts epoch training times from logs.
    Supports both '12.3s' and '2:23' formats.
    """
    if not os.path.exists(log_path):
        print(f"⚠️ Log file not found: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    times = []

    # Match "xx.xs" format
    for t in re.findall(r"(\d+\.\d+)s", text):
        times.append(float(t))

    # Match "m:ss" format (e.g., "2:23")
    for match in re.findall(r"(\d+):(\d+)", text):
        minutes, seconds = map(int, match)
        total_sec = minutes * 60 + seconds
        if total_sec < 3600:  # ignore large timestamps like hours
            times.append(total_sec)

    times = [t for t in times if t > 0]
    print(f"✅ Found {len(times)} epoch times in {os.path.basename(log_path)}")
    return times


# === 3️⃣ Extract times ===
cpu_times = extract_epoch_times(CPU_LOG)
gpu_times = extract_epoch_times(GPU_LOG)

if not cpu_times or not gpu_times:
    print("\n❌ Not enough data to plot. Please check your log files.")
    exit()

# === 4️⃣ Equalize epochs ===
min_len = min(len(cpu_times), len(gpu_times))
cpu_times = cpu_times[:min_len]
gpu_times = gpu_times[:min_len]
epochs = list(range(1, min_len + 1))

# === 5️⃣ Compute averages and speedup ===
cpu_avg = sum(cpu_times) / len(cpu_times)
gpu_avg = sum(gpu_times) / len(gpu_times)
speedup = cpu_avg / gpu_avg if gpu_avg else 0

print(f"\n📊 Average CPU epoch time: {cpu_avg:.2f} sec")
print(f"⚡ Average GPU epoch time: {gpu_avg:.2f} sec")
print(f"🚀 GPU Speedup: {speedup:.2f}×\n")

# === 6️⃣ Create line chart ===
plt.figure(figsize=(10, 6))
plt.plot(epochs, cpu_times, 'r-o', label="CPU")
plt.plot(epochs, gpu_times, 'g-o', label="GPU")
plt.xlabel("Epoch")
plt.ylabel("Training Time (seconds)")
plt.title("YOLOv8 Training Time per Epoch: CPU vs GPU")
plt.legend()
plt.grid(True)
plt.tight_layout()

line_plot_path = os.path.join(LOG_DIR, "training_time_comparison.png")
plt.savefig(line_plot_path)
print(f"📁 Line plot saved to: {line_plot_path}")

# === 7️⃣ Create bar chart ===
plt.figure(figsize=(6, 5))
plt.bar(["CPU", "GPU"], [cpu_avg, gpu_avg], color=["red", "green"])
plt.ylabel("Average Time per Epoch (s)")
plt.title("Average Training Time Comparison")
for i, v in enumerate([cpu_avg, gpu_avg]):
    plt.text(i, v + 5, f"{v:.1f}s", ha='center')

plt.tight_layout()

bar_plot_path = os.path.join(LOG_DIR, "average_time_comparison.png")
plt.savefig(bar_plot_path)
print(f"📁 Bar chart saved to: {bar_plot_path}")

plt.show()
