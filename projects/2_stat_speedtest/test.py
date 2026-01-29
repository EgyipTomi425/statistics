import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import time

csv_file = "column_stats_results.csv"
df = pd.read_csv(csv_file)

def python_column_stats(X):
    start = time.time()
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    skewness = skew(X, axis=0)
    kurt = kurtosis(X, axis=0, fisher=False)
    elapsed = (time.time() - start) * 1000
    return mean, var, skewness, kurt, elapsed

python_times = []
for idx, row in df.iterrows():
    rows = int(row['rows'])
    cols = int(row['cols'])
    X = np.random.uniform(-np.arange(1,cols+1), 2*np.arange(1,cols+1), size=(rows, cols))
    _, _, _, _, py_ms = python_column_stats(X)
    python_times.append(py_ms)

df['python_ms'] = python_times

def plot_times(df, fixed_dim='cols', filename=None):
    plt.figure(figsize=(12,6))
    if fixed_dim == 'cols':
        x = df['cols']
        xlabel = 'Columns'
    else:
        x = df['rows']
        xlabel = 'Rows'
    plt.plot(x, df['cpu_row_ms'], marker='o', label='CPU row-major (C++)')
    plt.plot(x, df['cpu_col_ms'], marker='o', label='CPU col-major (C++)')
    plt.plot(x, df['gpu_ms'], marker='o', label='GPU (C++)')
    plt.plot(x, df['python_ms'], marker='o', label='Python NumPy/Scipy')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel('Time (ms)')
    plt.title(f'Column stats timing vs {xlabel}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

df_rows_fixed = df[df['cols']==512]
df_cols_fixed = df[df['rows']==2**20]

plot_times(df_rows_fixed, fixed_dim='cols', filename='timing_rows.png')
plot_times(df_cols_fixed, fixed_dim='rows', filename='timing_cols.png')

df.to_csv("column_stats_results_with_python.csv", index=False)
