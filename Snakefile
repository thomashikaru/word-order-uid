
languages = ["en", "tr", "hu", "fr", "de", "ru", "vi", "id"]
variants = ["REAL_REAL", "REVERSE", "SORT_FREQ", "SORT_FREQ_REV", "MIN_DL_LOC", "MIN_DL_OPT", "RANDOM_1", "RANDOM_2", "RANDOM_3", "APPROX"]
parts = ["train", "test", "valid"]

RAW_DATA_DIR = "data/raw_data/wiki40b-txt"
SAMPLED_DATA_DIR = "data/raw_data/wiki40b-txt-sampled"
PARSE_DIR = "data/wiki40b-txt-parsed"
CF_DATA_DIR = "data/wiki40b-txt-cf"
CF_BPE_DATA_DIR = "data/wiki40b-txt-cf-bpe"
PREPROCESSED_DATA_DIR = "data/data-bin-cf-bpe"
CHECKPOINT_DIR = "data/checkpoint-cf-bpe"
EVAL_RESULTS_DIR = "data/perps-cf"
FASTBPE_PATH = "fastBPE/fast"
FASTBPE_NUM_OPS = 30000
FASTBPE_OUTPATH = "data/bpe_codes_cf/30k"

rule all:
    input:
        "data/plot1.png"

# sample and normalize wiki datasets
rule get_wiki40b_data:
    input:
    output:
        expand("data/raw_data/wiki40b-txt/{language}.{part}", language=languages, part=parts)
    shell: 
        """
    
        """

# sample and normalize wiki datasets
rule sample_wiki40b_data:
    input:
        "data/raw_data/wiki40b-txt/{language}.train",
        "data/raw_data/wiki40b-txt/{language}.valid",
        "data/raw_data/wiki40b-txt/{language}.test",
    output:
        "data/raw_data/wiki40b-txt-sampled/{language}.train",
        "data/raw_data/wiki40b-txt-sampled/{language}.valid",
        "data/raw_data/wiki40b-txt-sampled/{language}.test",
    shell: 
        """
    
        """

# convert sampled datasets into CONLLU dependency parses
rule do_dependency_parsing:
    input:
        "data/raw_data/wiki40b-txt-sampled/{language}.train",
        "data/raw_data/wiki40b-txt-sampled/{language}.valid",
        "data/raw_data/wiki40b-txt-sampled/{language}.test",
    output:
        "data/wiki40b-txt-parsed/{language}.train.conllu",
        "data/wiki40b-txt-parsed/{language}.valid.conllu",
        "data/wiki40b-txt-parsed/{language}.test.conllu",
    resources:
        time="24:00",
        num_cpus=1,
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
        logfile="logs_parse/log.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir {SAMPLED_DATA_DIR} --parse_dir {PARSE_DIR} --partitions 'train,test,valid'
        """.format(SAMPLED_DATA_DIR=SAMPLED_DATA_DIR, PARSE_DIR=PARSE_DIR)

# make counterfactual datsets for each language
rule make_cf_data:
    input:
        "data/wiki40b-txt-parsed/{language}.train.conllu",
        "data/wiki40b-txt-parsed/{language}.valid.conllu",
        "data/wiki40b-txt-parsed/{language}.test.conllu",
    output:
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.train",
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.valid",
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.test",
    resources:
        time="4:00",
        num_cpus=1,
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
        logfile="logs_cf/log.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        cd counterfactual
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.train.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.valid.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.test.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test
        """.format(CF_DATA_DIR=CF_DATA_DIR, PARSE_DIR=PARSE_DIR)

# train bpe on each dataset
rule train_bpe:
    input:
        "data/wiki40b-txt-cf/{language}/REAL_REAL/{language}.train"
    output:
        "data/bpe_codes/30k/{language}.codes"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd data
        cat {CF_DATA_DIR}/{{wildcards.language}}/*/{{wildcards.language}}.train | shuf > data/{{wildcards.language}}-agg.txt
        bsub -W 01:00 -n 1 -R "rusage[mem=2048,ngpus_excl_p=0]" \
            -o data/train_bpe_{{wildcards.language}}.out \
            {FASTBPE_PATH} learnbpe {FASTBPE_NUM_OPS} data/{{wildcards.language}}-agg.txt > {FASTBPE_OUTPATH}/{{wildcards.language}}.codes"
        """.format(CF_DATA_DIR=CF_DATA_DIR, FASTBPE_NUM_OPS=FASTBPE_NUM_OPS, FASTBPE_PATH=FASTBPE_PATH, FASTBPE_OUTPATH=FASTBPE_OUTPATH)

# apply the bpe to each dataset
rule apply_bpe:
    input:
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.train",
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.valid",
        "data/wiki40b-txt-cf/{language}/{variant}/{language}.test",
        "data/bpe_codes_cf/30k/{language}.codes"
    output:
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.train",
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.valid",
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.test",
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        bsub -W 01:00 -n 1 -R "rusage[mem=4000,ngpus_excl_p=0]" \
            -o data/apply_bpe_{{wildcards.language}}.out \
            {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        bsub -W 01:00 -n 1 -R "rusage[mem=4000,ngpus_excl_p=0]" \
            -o data/apply_bpe_{{wildcards.language}}.out \
            {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        bsub -W 01:00 -n 1 -R "rusage[mem=4000,ngpus_excl_p=0]" \
            -o data/apply_bpe_{{wildcards.language}}.out \
            {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR, FASTBPE_OUTPATH=FASTBPE_OUTPATH, FASTBPE_PATH=FASTBPE_PATH, CF_DATA_DIR=CF_DATA_DIR)

# binarize for fairseq training
rule prepare_fairseq_data:
    input:
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.train",
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.valid",
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.test",
    output:
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.train",
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.valid",
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.test"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        bsub -W 04:00 -n 1 -R "rusage[mem=8000,ngpus_excl_p=0]" \
            -o preprocess_data_cf.out \
            fairseq-preprocess \
                --only-source \
                --trainpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train \
                --validpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid \
                --testpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test \
                --destdir {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} \
                --bpe fastbpe \
                --workers 20 
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR, PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR)

# train the models
rule train_language_models:
    input:
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.train",
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.valid"
    output:
        "data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        bsub -W 24:00 -o logs-cf/{{wildcards.language}}-{{wildcards.variant}}.out \
            -R "select[gpu_mtotal0>=10000]"  \
            -R "rusage[mem=30000,ngpus_excl_p=1]" \
            bash train_model_transformer.sh {PREPROCESSED_DATA_DIR}/{{wildcards.variant}} {CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        """.format(PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR, CHECKPOINT_DIR=CHECKPOINT_DIR)

# evaluate the language models
rule eval_language_models:
    input:
        "data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt",
        "data/data-bin-cf-bpe/{language}/{variant}/{language}.test"
    output:
        "data/perps-cf/{language}-{variant}.pt"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        bsub -W 4:00 -o logs-cf-eval/{{wildcards.language}}-{{wildcards.variant}}.out \
            -R "select[gpu_mtotal0>=10000]"  \
            -R "rusage[mem=60000,ngpus_excl_p=1]" \
            python per_example_perp.py {CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}} {PREPROCESSED_DATA_DIR}/{{wildcards.variant}} {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {EVAL_RESULTS_DIR}/{{wildcards.language}}-{{wildcards.variant}}.pt
        """.format(CHECKPOINT_DIR=CHECKPOINT_DIR, PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR, CF_BPE_DATA_DIR=CF_BPE_DATA_DIR, EVAL_RESULTS_DIR=EVAL_RESULTS_DIR)

rule postprocess_eval_output:
    input:
        expand("data/perps-cf/{language}-{variant}.pt", language=languages, variant=variants)
    output:
        "data/eval_results_cf.feather"
    shell:
        """
    
        """

rule make_plots:
    input:
        "data/eval_results_cf.feather"
    output:
        "data/plot1.png"
    shell:
        """
    
        """