# Job details
TIME=01:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=0  # GPUs per node
# GPU_MODEL=GeForceGTX1080Ti  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=8000  # RAM for each core (default: 1024)

# OUTFILE=preprocess_data_cf.out  # default: lsf.oJOBID
OUTFILE=agg_unigram_freqs.out 

# Load modules
module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

bsub -W $TIME \
    -n $NUM_CPUS \
    -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
    -o $OUTFILE \
    python agg_unigram_freqs.py