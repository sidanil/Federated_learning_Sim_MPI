import pandas as pd
import matplotlib.pyplot as plt

def plot_speedup_efficiency(csv_path="timing_results.csv"):
    # Load results
    df = pd.read_csv(csv_path)

    # Extract serial baseline
    serial_times = df[df['mode'] == 'serial'].set_index('samples')['time']

    # Filter and compute speedup and efficiency
    mpi = df[df['mode'] == 'mpi'].copy()
    mpi['speedup'] = mpi.apply(lambda row: serial_times.get(row['samples'], None) / row['time'], axis=1)
    mpi['efficiency'] = mpi['speedup'] / mpi['ranks']

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Speedup plot
    for sample in sorted(mpi['samples'].unique()):
        subset = mpi[mpi['samples'] == sample]
        axs[0].plot(subset['ranks'], subset['speedup'], marker='o', label=f'{sample:,} samples')

    axs[0].set_title("Speedup vs Ranks (Varying Sample Sizes)")
    axs[0].set_xlabel("MPI Ranks")
    axs[0].set_ylabel("Speedup")
    axs[0].legend()
    axs[0].grid(True)

    # Efficiency plot
    for sample in sorted(mpi['samples'].unique()):
        subset = mpi[mpi['samples'] == sample]
        axs[1].plot(subset['ranks'], subset['efficiency'], marker='s', label=f'{sample:,} samples')

    axs[1].set_title("Efficiency vs Ranks (Varying Sample Sizes)")
    axs[1].set_xlabel("MPI Ranks")
    axs[1].set_ylabel("Efficiency")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("scaling_plots.png")
    plt.show()

plot_speedup_efficiency("timing_results.csv")
