FASTBPE=../fastBPE/fast  # path to the fastBPE tool
NUM_OPS=30000

# INPUT_DIR="wiki40b-txt-cf"
# OUTPATH=bpe_codes_cf/30k  # path where processed files will be stored

INPUT_DIR="wiki40b-txt-cf-v2"
OUTPATH=bpe_codes_cf_v2/30k

# Job details
TIME=01:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=0  # GPUs per node
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=2048  # RAM for each core (default: 1024)
 
# Load modules
module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

# create output path
mkdir -p $OUTPATH

# langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en" "de" "fr" "ru" "vi")

for lang in "${langlist[@]}"
do
    echo $lang
    cat $INPUT_DIR/$lang/*/$lang.train | shuf > $lang-agg.txt
    # learn bpe codes on the training set (or only use a subset of it)
    bsub -W $TIME \
                -n $NUM_CPUS \
                -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
                -o logs_cf_data/train_bpe_${lang}.out \
                $FASTBPE learnbpe $NUM_OPS $lang-agg.txt > $OUTPATH/$lang.codes
    # rm $lang-agg.txt
done
