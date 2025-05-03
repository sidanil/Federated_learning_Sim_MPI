#!/bin/bash

# --------- CONFIG ---------
MAX_RANKS=$(nproc)          # Detect number of logical CPU cores
SERIAL_SCRIPT="serial_federated_sim.py"
MPI_SCRIPT="mpi_federated_sim.py"
EPOCHS=100
ROUNDS=10
RESULTS_CSV="timing_results.csv"

# --------- HEADER ---------
echo "devices,mode,time" > $RESULTS_CSV

# --------- SERIAL RUNS ---------
for dev in 64
do
    if [ "$dev" -gt "$MAX_RANKS" ]; then continue; fi

    echo "Running SERIAL with $dev devices..."
    out=$(python $SERIAL_SCRIPT --total-devices $dev --epochs $EPOCHS --rounds $ROUNDS 2>/dev/null)
    time=$(echo "$out" | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$dev,serial,$time" >> $RESULTS_CSV
done

# --------- MPI RUNS ---------
for ranks in 1 2 4 8 16 32 64  
do
    if [ "$ranks" -gt "$MAX_RANKS" ]; then continue; fi

#    devices_per_rank=1  # fixed for now (can make this loop too)
#    total_devices=$((ranks * devices_per_rank))
    total_devices=64
    device_per_rank=$((total_devices/ranks))
    echo "Running MPI with $ranks ranks ($total_devices devices)..."
    out=$(mpiexec -n $ranks -- bash -c "python mpi_federated_sim.py --devices-per-rank $device_per_rank --epochs $EPOCHS --rounds $ROUNDS")
 
    time=$(echo "$out" | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$device_per_rank,mpi,$time" >> $RESULTS_CSV
done

echo "✅ Experiments complete. Results in $RESULTS_CSV"






