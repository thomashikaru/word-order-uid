# Job details
TIME=01:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=0  # GPUs per node
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=16000  # RAM for each core (default: 1024)
OUTFILE=apply_bpe_cf.out  # default: lsf.oJOBID

BPE_CODES=bpe_codes_cf/30k  # path where processed files will be stored
FASTBPE=../fastBPE/fast  # path to the fastBPE tool
INPUT_DIR="wiki40b-txt-cf"
OUT_DIR="wiki40b-txt-cf-bpe"

# Load modules
module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

# create output path
mkdir -p $OUT_DIR

# langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en" "de" "fr" "ru" "vi")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    echo $lang
    for ext in "${extlist[@]}"
    do
        # find all subdirectories within a language (should correspond to real, random, OV, and VO orderings)
        for D in $(find $INPUT_DIR/$lang -mindepth 1 -maxdepth 1 -type d)
        do
            model=$(basename $D)
            mkdir -p $OUT_DIR/$lang/$model
            bsub -W $TIME \
                -n $NUM_CPUS \
                -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
                -o $OUTFILE \
                $FASTBPE applybpe $OUT_DIR/$lang/$model/$lang.$ext $INPUT_DIR/$lang/$model/$lang.$ext  $BPE_CODES/$lang.codes
        done
    done
done

