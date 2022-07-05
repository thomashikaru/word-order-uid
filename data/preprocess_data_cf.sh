# Job details
TIME=04:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=0  # GPUs per node
# GPU_MODEL=GeForceGTX1080Ti  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=8000  # RAM for each core (default: 1024)

# OUTFILE=preprocess_data_cf.out  # default: lsf.oJOBID
OUTFILE=preprocess_data_cf_v2.out 

# Load modules
module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

train_pref="train"
valid_pref="valid"
test_pref="test"

# datadir="wiki40b-txt-cf-bpe"
# destdir="data-bin-cf-bpe"

datadir="wiki40b-txt-cf-bpe-v2"
destdir="data-bin-cf-bpe-v2"

for D in $(find $datadir -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D) 
    for M in $(find $datadir/$lang -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
	    mkdir -p $destdir/$lang/$model

        bsub -W $TIME \
            -n $NUM_CPUS \
            -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
            -o $OUTFILE \
            fairseq-preprocess \
                    --only-source \
                    --trainpref $datadir/$lang/$model/$lang.$train_pref \
                    --validpref $datadir/$lang/$model/$lang.$valid_pref \
                    --testpref $datadir/$lang/$model/$lang.$test_pref \
                    --destdir $destdir/$lang/$model \
                    --bpe fastbpe \
                    --workers 20
    done
done
