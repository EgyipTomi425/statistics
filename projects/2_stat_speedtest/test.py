import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import skew, kurtosis
import time
import os
from matplotlib.ticker import FuncFormatter

INPUT_CSV = "column_stats_results.csv"
OUTPUT_CSV = "column_stats_results_with_all.csv"
PICTURE_DIR = "pictures"
os.makedirs(PICTURE_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
use_cuda = torch.cuda.is_available()

print("Cuda status: " + str(use_cuda))

def numpy_stats(X):
    X = X.astype(np.float32)
    start = time.time()
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    skew_ = np.mean(((X - m)/s)**3, axis=0)
    kurt_ = np.mean(((X - m)/s)**4, axis=0)
    np.var(X, axis=0)
    return (time.time() - start) * 1000

def scipy_stats(X):
    X = X.astype(np.float32)
    start = time.time()
    np.mean(X, axis=0)
    np.var(X, axis=0)
    skew(X, axis=0)
    kurtosis(X, axis=0, fisher=False)
    return (time.time() - start) * 1000

def torch_stats(X, device='cpu'):
    t = torch.from_numpy(X.astype(np.float32)).to(device)
    if device=='cuda':
        torch.cuda.synchronize()
    start = time.time()
    m = torch.mean(t, dim=0)
    s = torch.std(t, dim=0, unbiased=False)
    skew_ = torch.mean(((t - m)/s)**3, dim=0)
    kurt_ = torch.mean(((t - m)/s)**4, dim=0)
    torch.mean(t, dim=0)
    torch.var(t, dim=0, unbiased=False)
    if device=='cuda':
        torch.cuda.synchronize()
    return (time.time() - start) * 1000

numpy_times = []
scipy_times = []
torch_cpu_times = []
torch_cuda_times = []

for _, row in df.iterrows():
    rows = int(row["rows"])
    cols = int(row["cols"])
    X = np.random.uniform(-np.arange(1, cols + 1), 2 * np.arange(1, cols + 1), size=(rows, cols)).astype(np.float32)
    numpy_times.append(numpy_stats(X))
    scipy_times.append(scipy_stats(X))
    torch_cpu_times.append(torch_stats(X, device='cpu'))
    torch_cuda_times.append(torch_stats(X, device='cuda') if use_cuda else np.nan)

df["numpy_ms"] = numpy_times
df["scipy_ms"] = scipy_times
df["torch_cpu_ms"] = torch_cpu_times
df["torch_cuda_ms"] = torch_cuda_times

df.to_csv(OUTPUT_CSV, index=False)

first_cols_value = int(round(df.iloc[0]["cols"]))
df_first_part = df[df["cols"] == df.iloc[0]["cols"]]

plt.figure(figsize=(10,6))
plt.plot(df_first_part["rows"], df_first_part["cpu_row_ms"], marker="o", label="CPU row-major", color="#1f77b4")
plt.plot(df_first_part["rows"], df_first_part["cpu_col_ms"], marker="o", label="CPU col-major", color="#6fffff")
plt.plot(df_first_part["rows"], df_first_part["gpu_ms"], marker="o", label="CUDA", color="#004400")
plt.plot(df_first_part["rows"], df_first_part["numpy_ms"], marker="o", label="NumPy", color="#d62728")
plt.plot(df_first_part["rows"], df_first_part["scipy_ms"], marker="o", label="SciPy", color="#9467bd")
plt.plot(df_first_part["rows"], df_first_part["torch_cpu_ms"], marker="o", label="Torch CPU", color="orange")
if use_cuda:
    plt.plot(df_first_part["rows"], df_first_part["torch_cuda_ms"], marker="o", label="Torch CUDA", color="#66ff66")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Rows")
plt.ylabel("Time (ms)")
plt.title(f"Column statistics timing ({first_cols_value} columns)")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.legend()
plt.savefig(f"{PICTURE_DIR}/timing_{first_cols_value}_columns.png", dpi=300)
plt.show()

last_rows_value = int(round(df.iloc[-1]["rows"]))
df_second_part = df[df["rows"] == df.iloc[-1]["rows"]].sort_values("cols")

plt.figure(figsize=(10,6))
plt.plot(df_second_part["cols"], df_second_part["cpu_row_ms"], marker="o", label="CPU row-major", color="#1f77b4")
plt.plot(df_second_part["cols"], df_second_part["cpu_col_ms"], marker="o", label="CPU col-major", color="#6fffff")
plt.plot(df_second_part["cols"], df_second_part["gpu_ms"], marker="o", label="CUDA", color="#004400")
plt.plot(df_second_part["cols"], df_second_part["numpy_ms"], marker="o", label="NumPy", color="#d62728")
plt.plot(df_second_part["cols"], df_second_part["scipy_ms"], marker="o", label="SciPy", color="#9467bd")
plt.plot(df_second_part["cols"], df_second_part["torch_cpu_ms"], marker="o", label="Torch CPU", color="orange")
if use_cuda:
    plt.plot(df_second_part["cols"], df_second_part["torch_cuda_ms"], marker="o", label="Torch CUDA", color="#66ff66")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Columns")
plt.ylabel("Time (ms)")
plt.title(f"Column statistics timing ({last_rows_value} rows)")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.legend()
plt.savefig(f"{PICTURE_DIR}/timing_{last_rows_value}_rows.png", dpi=300)
plt.show()