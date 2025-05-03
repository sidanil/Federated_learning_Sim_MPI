#!/bin/bash

# --------- CONFIG ---------
MAX_RANKS=$(nproc)           # Detect number of logical CPU cores
SERIAL_SCRIPT="serial_federated_sim.py"
MPI_SCRIPT="mpi_federated_sim.py"
EPOCHS=100
ROUNDS=10
RESULTS_CSV="timing_results.csv"
SAMPLE_SIZES=(100000 500000 1000000 5000000 10000000)
TOTAL_DEVICES=64
RANKS_LIST=(1 2 4 8 16 32 64)

# --------- HEADER ---------
if [ -f "$RESULTS_CSV" ]; then
    echo "⚠️  File $RESULTS_CSV exists. Overwriting..."
    rm "$RESULTS_CSV"
fi
echo "samples,ranks,devices,time,mode" > $RESULTS_CSV

# --------- RUN EXPERIMENTS ---------
for SAMPLES in "${SAMPLE_SIZES[@]}"
do
    echo "▶️ Running experiments for $SAMPLES samples..."

    # --- SERIAL ---
    echo "Running SERIAL with $TOTAL_DEVICES devices..."
    out=$(python $SERIAL_SCRIPT --total-devices $TOTAL_DEVICES --epochs $EPOCHS --rounds $ROUNDS --samples $SAMPLES 2>/dev/null)
    time=$(echo "$out" | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$SAMPLES,1,$TOTAL_DEVICES,$time,serial" >> $RESULTS_CSV

    # --- MPI ---
    for RANKS in "${RANKS_LIST[@]}"
    do
        if (( TOTAL_DEVICES % RANKS != 0 )); then
            echo "Skipping: $TOTAL_DEVICES not divisible by $RANKS ranks"
            continue
        fi

        if [ "$RANKS" -gt "$MAX_RANKS" ]; then
            echo "Skipping: $RANKS ranks exceeds available cores ($MAX_RANKS)"
            continue
        fi

        DEVICES_PER_RANK=$((TOTAL_DEVICES / RANKS))
        echo "Running MPI with $RANKS ranks ($DEVICES_PER_RANK devices/rank)..."
        out=$(mpiexec -n $RANKS -- bash -c "python $MPI_SCRIPT --devices-per-rank $DEVICES_PER_RANK --epochs $EPOCHS --rounds $ROUNDS --samples $SAMPLES" 2>/dev/null)
        time=$(echo "$out" | grep -oP 'Time:\s+\K[0-9.]+')

        if [ -n "$time" ]; then
            echo "$SAMPLES,$RANKS,$TOTAL_DEVICES,$time,mpi" >> $RESULTS_CSV
        else
            echo "❌ MPI failed for $RANKS ranks, $SAMPLES samples"
        fi
    done
done

echo "✅ All experiments complete. Results saved to $RESULTS_CSV"
