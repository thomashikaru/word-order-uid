
languages = ["en", "tr", "hu", "fr", "de", "ru", "vi", "id", "hi", "fa"]
languages_100m = ["en"]
languages_cc100 = ["en", "tr", "hi", "ru", "hu"]
variants = ["REAL_REAL", "REVERSE", "SORT_FREQ", "SORT_FREQ_REV", "MIN_DL_PROJ", "MIN_DL_OPT", "RANDOM_1", "RANDOM_2", "RANDOM_3", "RANDOM_4", "RANDOM_5", "APPROX", "EFFICIENT_VO", "EFFICIENT_OV"]
parts = ["train", "test", "valid"]

BASE_DIR = "/cluster/work/cotterell/tclark/word-order-uid"
RAW_DATA_DIR = "data/raw_data/wiki40b-txt"
RAW_DATA_DIR_cc100 = "data/cc100/txt"
SAMPLED_DATA_DIR = "data/raw_data/wiki40b-txt-sampled"
SAMPLED_DATA_DIR_100m = "data/raw_data/wiki40b-txt-sampled-100m"
SAMPLED_DATA_DIR_cc100 = "data/cc100/txt-sampled"
PARSE_DIR = "data/wiki40b-txt-parsed"
PARSE_DIR_100m = "data/wiki40b-txt-parsed-100m"
PARSE_DIR_cc100 = "data/cc100/txt-parsed"
CF_DATA_DIR = "data/wiki40b-txt-cf"
CF_DATA_DIR_100m = "data/wiki40b-txt-cf-100m"
CF_DATA_DIR_cc100 = "data/cc100/txt-cf"
CF_BPE_DATA_DIR = "data/wiki40b-txt-cf-bpe"
CF_BPE_DATA_DIR_100m = "data/wiki40b-txt-cf-bpe-100m"
CF_BPE_DATA_DIR_cc100 = "data/cc100/txt-cf-bpe"
PREPROCESSED_DATA_DIR = "data/data-bin-cf-bpe"
PREPROCESSED_DATA_DIR_100m = "data/data-bin-cf-bpe-100m"
PREPROCESSED_DATA_DIR_cc100 = "data/cc100/data-bin-cf-bpe"
CHECKPOINT_DIR = "data/checkpoint-cf-bpe"
CHECKPOINT_DIR_100m = "data/checkpoint-cf-bpe-100m"
CHECKPOINT_DIR_cc100 = "data/cc100/checkpoint-cf-bpe"
EVAL_RESULTS_DIR = "evaluation/perps-cf"
EVAL_RESULTS_DIR_100m = "evaluation/perps-cf-100m"
EVAL_RESULTS_DIR_cc100 = "evaluation/cc100/perps-cf"
FASTBPE_PATH = "fastBPE/fast"
FASTBPE_NUM_OPS = 30000
FASTBPE_OUTPATH = "data/bpe_codes_cf/30k"

rule all:
    input:
        "evaluation/plots/joint_surprisal_and_variance.png",
        "evaluation/eval_results_cf_100m.feather"

# sample and normalize wiki datasets
rule get_wiki40b_data:
    input:
    output:
        expand("data/raw_data/wiki40b-txt/{language}.{part}", language=languages, part=parts)
    shell: 
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd data
        python wiki_40b.py --lang_code_list {{wildcards.language}} --data_dir "tfdata" --output_prefix "raw_data/wiki40b-txt/"
        """

rule get_cc100_data:
    input:
    output:
        expand("data/cc100/txt/{{language}}.txt")
    resources:
        time="8:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_get_data_{language}_cc100.out"
    shell:
        """
        module load python_gpu/3.8.5 hdf5 eth_proxy
        mkdir -p data/cc100/txt
        cd data/cc100/txt
        wget https://data.statmt.org/cc-100/{wildcards.language}.txt.xz
        unxz {wildcards.language}.txt.xz
        """

# sample wiki40b datasets (~20M tokens per language)
rule sample_wiki40b_data:
    input:
        expand("data/raw_data/wiki40b-txt/{{language}}.{part}", part=parts)
    output:
        expand("data/raw_data/wiki40b-txt-sampled/{{language}}.{part}", part=parts)
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_sample_{language}_100m.out"
    shell: 
        f"""
        cd data
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {RAW_DATA_DIR} --output_prefix {SAMPLED_DATA_DIR} --wiki40b
        """

# sample wiki40b datasets (~100M tokens for en-large)
rule sample_wiki40b_data_100m:
    input:
        expand("data/raw_data/wiki40b-txt/{{language}}.{part}", part=parts)
    output:
        expand("data/raw_data/wiki40b-txt-sampled-100m/{{language}}.{part}", part=parts)
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_sample_{language}_100m.out"
    shell: 
        f"""
        mkdir -p data/raw_data/wiki40b-txt-sampled-100m
        cd data
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {RAW_DATA_DIR} --output_prefix {SAMPLED_DATA_DIR_100m} --num_train_tokens 100000000 --wiki40b
        """

# sample cc100 datasets (~20M tokens per language)
rule sample_cc100:
    input:
        expand("data/cc100/txt/{{language}}.txt", part=parts)
    output:
        expand("data/cc100/txt-sampled/{{language}}.{part}", part=parts)
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_sample_{language}_cc100.out"
    shell: 
        f"""
        mkdir -p {SAMPLED_DATA_DIR_cc100}
        cd data
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {BASE_DIR}/{RAW_DATA_DIR_cc100} --output_prefix {BASE_DIR}/{SAMPLED_DATA_DIR_cc100} --num_train_tokens 20000000 --cc100
        """

# convert sampled datasets into CONLLU dependency parses
rule do_dependency_parsing:
    input:
        expand("data/raw_data/wiki40b-txt-sampled/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-parsed/{{language}}.{part}.conllu", part=parts)
    resources:
        time="36:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_parse_{language}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR} --parse_dir ../{PARSE_DIR} --partitions 'train,test,valid'
        """.format(SAMPLED_DATA_DIR=SAMPLED_DATA_DIR, PARSE_DIR=PARSE_DIR)

# convert sampled datasets into CONLLU dependency parses
rule do_dependency_parsing_100m:
    input:
        expand("data/raw_data/wiki40b-txt-sampled-100m/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-parsed-100m/{{language}}.{part}.conllu", part=parts)
    resources:
        time="36:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_parse_{language}_100m.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR} --parse_dir ../{PARSE_DIR} --partitions 'train,test,valid'
        """.format(SAMPLED_DATA_DIR=SAMPLED_DATA_DIR_100m, PARSE_DIR=PARSE_DIR_100m)

# convert sampled datasets into CONLLU dependency parses
rule do_dependency_parsing_cc100:
    input:
        expand("data/cc100/txt-sampled/{{language}}.{part}", part=parts)
    output:
        expand("data/cc100/txt-parsed/{{language}}.{part}.conllu", part=parts)
    resources:
        time="36:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_parse_{language}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR_cc100}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR_cc100} --parse_dir ../{PARSE_DIR_cc100} --partitions 'train,test,valid'
        """

# convert sampled datasets into CONLLU dependency parses
rule do_dependency_parsing_test_run:
    input:
        expand("data/raw_data/wiki40b-txt-sampled/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-parsed/{{language}}.{part}.tiny.conllu", part=parts)
    resources:
        time="24:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_parse_{language}_test_run.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR} --parse_dir ../{PARSE_DIR} --partitions 'train,test,valid' --test_run
        """.format(SAMPLED_DATA_DIR=SAMPLED_DATA_DIR, PARSE_DIR=PARSE_DIR)

# convert sampled datasets into CONLLU dependency parses
rule get_unigram_freqs:
    input:
        "data/raw_data/wiki40b-txt-sampled/{language}.train",
    output:
        "counterfactual/freqs/{language}.csv"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_unigram_freqs_{language}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd counterfactual
        python save_unigram_freqs.py --langs {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR}
        """.format(SAMPLED_DATA_DIR=SAMPLED_DATA_DIR)

# make counterfactual datsets for each language
rule make_cf_data:
    input:
        expand("data/wiki40b-txt-parsed/{{language}}.{part}.conllu", part=parts),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/wiki40b-txt-cf/{{language}}/{{variant}}/{{language}}.{part}", part=parts)
    resources:
        time="08:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_cf_{language}_{variant}.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        cd counterfactual
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.train.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.valid.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.test.conllu > ../{CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test
        """

# make counterfactual datsets for each language
rule make_cf_data_100m:
    input:
        expand("data/wiki40b-txt-parsed-100m/{{language}}.{part}.conllu", part=parts),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/wiki40b-txt-cf-100m/{{language}}/{{variant}}/{{language}}.{part}", part=parts)
    resources:
        time="24:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_cf_{language}_{variant}_100m.out"
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
        """.format(CF_DATA_DIR=CF_DATA_DIR_100m, PARSE_DIR=PARSE_DIR_100m)

# make counterfactual datsets for each language
rule make_cf_data_cc100:
    input:
        expand("data/cc100/txt-parsed/{{language}}.{part}.conllu", part=parts),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/cc100/txt-cf/{{language}}/{{variant}}/{{language}}.{part}", part=parts)
    resources:
        time="24:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_cf_{language}_{variant}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        cd counterfactual
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR_cc100}/{{wildcards.language}}.train.conllu > ../{CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR_cc100}/{{wildcards.language}}.valid.conllu > ../{CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR_cc100}/{{wildcards.language}}.test.conllu > ../{CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test
        """

rule all_cf_data_wiki40b_20m:
    input:
        expand("data/wiki40b-txt-cf/{language}/{variant}/{language}.{part}", language=languages, variant=variants, part=parts),

rule all_cf_data_cc100:
    input:
        expand("data/cc100/txt-cf/{language}/{variant}/{language}.{part}", language=languages_cc100, variant=variants, part=parts),

# train bpe on each dataset
rule train_bpe:
    input:
        "data/wiki40b-txt-cf/{language}/REAL_REAL/{language}.train"
    output:
        "data/bpe_codes_cf/30k/{language}.codes"
    resources:
        time="01:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=16000,ngpus_excl_p=0]",
    log: 
        "data/logs_thc/log_train_bpe_{language}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p data/bpe_codes_cf/30k
        cat {CF_DATA_DIR}/{{wildcards.language}}/*/{{wildcards.language}}.train | shuf | head -n 100000 > data/{{wildcards.language}}-agg.txt
        {FASTBPE_PATH} learnbpe {FASTBPE_NUM_OPS} data/{{wildcards.language}}-agg.txt > {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """.format(CF_DATA_DIR=CF_DATA_DIR, FASTBPE_NUM_OPS=FASTBPE_NUM_OPS, FASTBPE_PATH=FASTBPE_PATH, FASTBPE_OUTPATH=FASTBPE_OUTPATH)

# apply the bpe to each dataset
rule apply_bpe:
    input:
        expand("data/wiki40b-txt-cf/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
        "data/bpe_codes_cf/30k/{language}.codes"
    output:
        expand("data/wiki40b-txt-cf-bpe/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    resources:
        time="01:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_apply_bpe_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR, FASTBPE_OUTPATH=FASTBPE_OUTPATH, FASTBPE_PATH=FASTBPE_PATH, CF_DATA_DIR=CF_DATA_DIR)

rule apply_bpe_100m:
    input:
        expand("data/wiki40b-txt-cf-100m/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
        "data/bpe_codes_cf/30k/{language}.codes"
    output:
        expand("data/wiki40b-txt-cf-bpe-100m/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_apply_bpe_{language}_{variant}_100m.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR_100m, FASTBPE_OUTPATH=FASTBPE_OUTPATH, FASTBPE_PATH=FASTBPE_PATH, CF_DATA_DIR=CF_DATA_DIR_100m)

rule apply_bpe_cc100:
    input:
        expand("data/cc100/txt-cf/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
        "data/bpe_codes_cf/30k/{language}.codes"
    output:
        expand("data/cc100/txt-cf-bpe/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_apply_bpe_{language}_{variant}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {CF_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """

rule apply_all_bpe_wiki40b:
    input:
        expand("data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.{part}", language=languages, variant=variants, part=parts)

# binarize for fairseq training
rule prepare_fairseq_data:
    input:
        expand("data/wiki40b-txt-cf-bpe/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    output:
        expand("data/data-bin-cf-bpe/{{language}}/{{variant}}/{part}.bin", part=parts),
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_preprocess_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        rm -r {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        mkdir -p {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        fairseq-preprocess \
            --only-source \
            --trainpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train \
            --validpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid \
            --testpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test \
            --destdir {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} \
            --bpe fastbpe \
            --workers 20 
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR, PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR)

# binarize for fairseq training
rule prepare_fairseq_data_100m:
    input:
        expand("data/wiki40b-txt-cf-bpe-100m/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    output:
        expand("data/data-bin-cf-bpe-100m/{{language}}/{{variant}}/{part}.bin", part=parts),
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_preprocess_{language}_{variant}_100m.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        rm -r {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        mkdir -p {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        fairseq-preprocess \
            --only-source \
            --trainpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train \
            --validpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid \
            --testpref {CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test \
            --destdir {PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} \
            --bpe fastbpe \
            --workers 20 
        """.format(CF_BPE_DATA_DIR=CF_BPE_DATA_DIR_100m, PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR_100m)

# binarize for fairseq training
rule prepare_fairseq_data_cc100:
    input:
        expand("data/cc100/txt-cf-bpe/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    output:
        expand("data/cc100/data-bin-cf-bpe/{{language}}/{{variant}}/{part}.bin", part=parts),
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_preprocess_{language}_{variant}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        rm -r {PREPROCESSED_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        mkdir -p {PREPROCESSED_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        fairseq-preprocess \
            --only-source \
            --trainpref {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train \
            --validpref {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid \
            --testpref {CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test \
            --destdir {PREPROCESSED_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}} \
            --bpe fastbpe \
            --workers 20 
        """

rule all_dataset_prep:
    input:
        expand("data/data-bin-cf-bpe/{language}/{variant}/{part}.bin", language=languages, variant=variants, part=parts),
        expand("data/data-bin-cf-bpe-100m/{language}/{variant}/{part}.bin", language=languages_100m, variant=variants, part=parts),
        expand("data/data-bin-cf-bpe-cc100/{language}/{variant}/{part}.bin", language=languages_cc100, variant=variants, part=parts)

rule all_dataset_prep_wiki40b:
    input:
        expand("data/data-bin-cf-bpe/{language}/{variant}/{part}.bin", language=languages, variant=variants, part=parts)

rule all_dataset_prep_cc100:
    input:
        expand("data/cc100/data-bin-cf-bpe/{language}/{variant}/{part}.bin", language=languages_cc100, variant=variants, part=parts)

# train the models
rule train_language_models:
    input:
        "data/data-bin-cf-bpe/{language}/{variant}/train.bin",
        "data/data-bin-cf-bpe/{language}/{variant}/valid.bin",
    output:
        "data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt"
    resources:
        time="24:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_train_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        cd data
        bash train_model_transformer.sh ../{PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} \
            ../{CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        """.format(PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR, CHECKPOINT_DIR=CHECKPOINT_DIR)

# train the models
rule train_language_models_100m:
    input:
        "data/data-bin-cf-bpe-100m/{language}/{variant}/train.bin",
        "data/data-bin-cf-bpe-100m/{language}/{variant}/valid.bin",
    output:
        "data/checkpoint-cf-bpe-100m/{language}/{variant}/checkpoint_best.pt"
    resources:
        time="96:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_train_{language}_{variant}_100m.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        cd data
        bash train_model_transformer.sh ../{PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} \
            ../{CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        """.format(PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR_100m, CHECKPOINT_DIR=CHECKPOINT_DIR_100m)

# train the models
rule train_language_models_cc100:
    input:
        "data/cc100/data-bin-cf-bpe/{language}/{variant}/train.bin",
        "data/cc100/data-bin-cf-bpe/{language}/{variant}/valid.bin",
    output:
        "data/cc100/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt"
    resources:
        time="24:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_train_{language}_{variant}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CHECKPOINT_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        cd data
        bash train_model_transformer.sh ../{PREPROCESSED_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}} \
            ../{CHECKPOINT_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}
        """

rule all_models_train_wiki40b:
    input: 
        expand("data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt", language=languages, variant=variants)

rule all_models_train_cc100:
    input: 
        expand("data/cc100/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt", language=languages_cc100, variant=variants)

rule all_models_train_wiki40b_hi_fa:
    input: 
        expand("data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt", language=["hi", "fa"], variant=variants)

# evaluate the language models
rule eval_language_models:
    input:
        "data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt",
        "data/wiki40b-txt-cf-bpe/{language}/{variant}/{language}.test",
        "data/data-bin-cf-bpe/{language}/{variant}/test.bin"
    output:
        "evaluation/perps-cf/{language}-{variant}.pt"
    resources:
        time="4:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_eval_{language}_{variant}.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {EVAL_RESULTS_DIR}
        cd data
        python per_example_perp.py {BASE_DIR}/{CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {BASE_DIR}/{EVAL_RESULTS_DIR}/{{wildcards.language}}-{{wildcards.variant}}.pt
        """

rule eval_language_models_wiki40b:
    input:
        expand("evaluation/perps-cf/{language}-{variant}.pt", language=languages, variant=variants)

rule eval_language_models_hi_fa:
    input:
        expand("evaluation/perps-cf/{language}-{variant}.pt", language=["hi", "fa"], variant=variants)

# evaluate the language models
rule eval_language_models_100m:
    input:
        "data/checkpoint-cf-bpe-100m/{language}/{variant}/checkpoint_best.pt",
        "data/wiki40b-txt-cf-bpe-100m/{language}/{variant}/{language}.test",
        "data/data-bin-cf-bpe-100m/{language}/{variant}/test.bin"
    output:
        "evaluation/perps-cf-100m/{language}-{variant}.pt"
    resources:
        time="4:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_eval_{language}_{variant}_100m.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {EVAL_RESULTS_DIR}
        cd data
        python per_example_perp.py {BASE_DIR}/{CHECKPOINT_DIR}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{PREPROCESSED_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{CF_BPE_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {BASE_DIR}/{EVAL_RESULTS_DIR}/{{wildcards.language}}-{{wildcards.variant}}.pt
        """.format(BASE_DIR=BASE_DIR, CHECKPOINT_DIR=CHECKPOINT_DIR_100m, PREPROCESSED_DATA_DIR=PREPROCESSED_DATA_DIR_100m, CF_BPE_DATA_DIR=CF_BPE_DATA_DIR_100m, EVAL_RESULTS_DIR=EVAL_RESULTS_DIR_100m)

rule eval_language_models_100m_all:
    input:
        expand("evaluation/perps-cf-100m/{language}-{variant}.pt", language=languages_100m, variant=variants)

# evaluate the language models
rule eval_language_models_cc100:
    input:
        "data/cc100/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt",
        "data/cc100/txt-cf-bpe/{language}/{variant}/{language}.test",
        "data/cc100/data-bin-cf-bpe/{language}/{variant}/test.bin"
    output:
        "evaluation/cc100/perps-cf/{language}-{variant}.pt"
    resources:
        time="4:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_eval_{language}_{variant}_cc100.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {EVAL_RESULTS_DIR_cc100}
        cd data
        python per_example_perp.py {BASE_DIR}/{CHECKPOINT_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{PREPROCESSED_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{CF_BPE_DATA_DIR_cc100}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {BASE_DIR}/{EVAL_RESULTS_DIR_cc100}/{{wildcards.language}}-{{wildcards.variant}}.pt
        """

rule eval_language_models_cc100_all:
    input:
        expand("evaluation/cc100/perps-cf/{language}-{variant}.pt", language=languages_cc100, variant=variants)

rule postprocess_eval_output:
    input:
        expand("evaluation/perps-cf/{language}-{variant}.pt", language=languages, variant=variants)
    output:
        "evaluation/eval_results_cf.feather"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=32000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_postprocess_eval_output.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python evaluation.py --make_csv --perps_file_pattern 'perps-cf/*.pt' --out_file 'eval_results_cf.feather'
        """

rule postprocess_eval_output_100m:
    input:
        expand("evaluation/perps-cf-100m/{language}-{variant}.pt", language=languages_100m, variant=variants)
    output:
        "evaluation/eval_results_cf_100m.feather"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python evaluation.py --make_csv --perps_file_pattern 'perps-cf-100m/*.pt' --out_file 'eval_results_cf_100m.feather'
        """

rule postprocess_eval_output_cc100:
    input:
        expand("evaluation/cc100/perps-cf/{language}-{variant}.pt", language=languages_cc100, variant=variants)
    output:
        "evaluation/cc100/eval_results_cf_cc100.feather"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python evaluation.py --make_csv --perps_file_pattern 'cc100/perps-cf/*.pt' --out_file 'cc100/eval_results_cf_cc100.feather'
        """

rule make_plotting_inputs:
    input:
        "evaluation/eval_results_cf.feather"
    output:
        "evaluation/plot_csv/{uid_metric}.csv",
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=24000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_make_plotting_inputs_{uid_metric}.out"
    shell:
        """
        cd evaluation
        mkdir -p plot_csv
        python make_plotting_inputs.py --inputfile eval_results_cf.feather --data_dir plot_csv --metric {wildcards.uid_metric}
        """


uid_metrics = [
    "surprisal",
    "surprisal_variance",
    "doc_initial_var",
    "surprisal_deviations",
    "delta_surps",
    "infs_1.1",
    "infs_1.25",
    "avg_surps",
    "delta_surps_by_tok",
    "max_surps"]

rule make_plotting_inputs_wiki40b:
    input:
        expand("evaluation/plot_csv/{uid_metric}.csv", uid_metric=uid_metrics)


rule wiki40b_make_plotting_inputs:
    input:
        "evaluation/plot_csv/surprisal_plot_vals.csv",
        "evaluation/plot_csv/surprisal_variance_plot_vals.csv",
        "evaluation/plot_csv/doc_initial_var.csv",
        "evaluation/plot_csv/surprisal_deviations_plot_vals.csv",
        "evaluation/plot_csv/delta_surps_plot_vals.csv",
        "evaluation/plot_csv/infs_1.1_plot_vals.csv",
        "evaluation/plot_csv/infs_plot_vals.csv",
        "evaluation/plot_csv/avg_surps_plot_vals.csv",
        "evaluation/plot_csv/delta_surps_by_tok.csv",
        "evaluation/plot_csv/max_surps_plot_vals.csv",

rule make_plotting_inputs_100m:
    input:
        "evaluation/eval_results_cf_100m.feather"
    output:
        "evaluation/plot_csv_100m/surprisal_plot_vals.csv",
        "evaluation/plot_csv_100m/surprisal_variance_plot_vals.csv",
        "evaluation/plot_csv_100m/doc_initial_var.csv",
        "evaluation/plot_csv_100m/surprisal_deviations_plot_vals.csv",
        "evaluation/plot_csv_100m/delta_surps_plot_vals.csv",
        "evaluation/plot_csv_100m/infs_1.1_plot_vals.csv",
        "evaluation/plot_csv_100m/infs_plot_vals.csv",
        "evaluation/plot_csv_100m/avg_surps_plot_vals.csv",
        "evaluation/plot_csv_100m/delta_surps_by_tok.csv",
        "evaluation/plot_csv_100m/max_surps_plot_vals.csv",
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    shell:
        """
        cd evaluation
        mkdir -p plot_csv_100m
        python make_plotting_input.py --inputfile eval_results_cf_100m.feather --data_dir plot_csv_100m
        """

rule make_plotting_inputs_cc100:
    input:
        "evaluation/cc100/eval_results_cf_cc100.feather"
    output:
        "evaluation/cc100/plot_csv/surprisal_plot_vals.csv",
        "evaluation/cc100/plot_csv/surprisal_variance_plot_vals.csv",
        "evaluation/cc100/plot_csv/doc_initial_var.csv",
        "evaluation/cc100/plot_csv/surprisal_deviations_plot_vals.csv",
        "evaluation/cc100/plot_csv/delta_surps_plot_vals.csv",
        "evaluation/cc100/plot_csv/infs_1.1_plot_vals.csv",
        "evaluation/cc100/plot_csv/infs_plot_vals.csv",
        "evaluation/cc100/plot_csv/avg_surps_plot_vals.csv",
        "evaluation/cc100/plot_csv/delta_surps_by_tok.csv",
        "evaluation/cc100/plot_csv/max_surps_plot_vals.csv",
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    shell:
        """
        cd evaluation
        mkdir -p cc100/plot_csv
        python make_plotting_input.py --inputfile cc100/eval_results_cf_cc100.feather --data_dir cc100/plot_csv
        """

rule make_plots:
    input:
        "evaluation/plot_csv/surprisal_plot_vals.csv",
        "evaluation/plot_csv/surprisal_variance_plot_vals.csv",
        "evaluation/plot_csv/doc_initial_var.csv",
        "evaluation/plot_csv/surprisal_deviations_plot_vals.csv",
        "evaluation/plot_csv/delta_surps_plot_vals.csv",
        "evaluation/plot_csv/infs_1.1_plot_vals.csv",
        "evaluation/plot_csv/infs_plot_vals.csv",
        "evaluation/plot_csv/avg_surps_plot_vals.csv",
        "evaluation/plot_csv/delta_surps_by_tok.csv",
        "evaluation/plot_csv/max_surps_plot_vals.csv",
    output:
        "evaluation/plots/joint_surprisal_and_variance.png",
        "evaluation/plots/joint_mean_regress_and_uid_power.png",
        "evaluation/plots/joint_doc_initial_and_uid_loc.png",
        "evaluation/plots/surprisal_variance_dataset_mean_point.png",
        "evaluation/plots/surprisal_variance_doc_initial_point.png",
        "evaluation/plots/delta_surp_point.png",
        "evaluation/plots/max_surp_point.png",
        "evaluation/plots/uid_power_1.25_point.png",
        "evaluation/plots/uid_power_1.1_point.png",
        "evaluation/plots/surprisal_by_token_position.png",
        "evaluation/plots/delta_surprisal_by_token_position.png"
    shell:
        """
        cd evaluation
        mkdir -p plots
        Rscript tacl_plots.R
        """



### Measuring Dependency Length

rule measure_dl:
    input:
        expand("data/wiki40b-txt-parsed/{{language}}.test.conllu"),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/wiki40b-txt-cf-deplens/{{language}}/{{variant}}/testset_deplens.txt")
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_measure_dl_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p data/wiki40b-txt-cf-deplens/{{wildcards.language}}/{{wildcards.variant}}
        cd counterfactual
        python apply_counterfactual_grammar.py --output_dl_only --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.language}}.test.conllu > ../data/wiki40b-txt-cf-deplens/{{wildcards.language}}/{{wildcards.variant}}/testset_deplens.txt
        """.format(PARSE_DIR=PARSE_DIR)

rule measure_dl_wiki40b:
    input:
        expand("data/wiki40b-txt-cf-deplens/{language}/{variant}/testset_deplens.txt", language=languages, variant=variants)



### making datasets of different sizes

# sample wiki40b datasets
rule sample_wiki40b_data_diff_sizes:
    input:
        expand("data/raw_data/wiki40b-txt/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-sampled-diff-sizes/{{num_toks}}/{{language}}.{part}", part=parts)
    resources:
        time="12:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_sample_{language}_{num_toks}.out"
    shell: 
        f"""
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {RAW_DATA_DIR} --output_prefix data/wiki40b-txt-sampled-diff-sizes --num_train_tokens {{wildcards.num_toks}}
        """

rule do_dependency_parsing_diff_sizes:
    input:
        expand("data/wiki40b-txt-sampled-diff-sizes/{{num_toks}}/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-parsed-diff-sizes/{{num_toks}}/{{language}}.{part}.conllu", part=parts)
    resources:
        time="24:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_parse_{language}_{num_toks}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {PARSE_DIR}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir ../{SAMPLED_DATA_DIR} --parse_dir ../{PARSE_DIR} --partitions 'train,test,valid'
        """.format(SAMPLED_DATA_DIR="wiki40b-txt-sampled-diff-sizes", PARSE_DIR="wiki40b-txt-parsed-diff-sizes")

# make counterfactual datsets for each language
rule make_cf_data_diff_sizes:
    input:
        expand("data/wiki40b-txt-parsed-diff-sizes/{{num_toks}}/{{language}}.{part}.conllu", part=parts),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/wiki40b-txt-cf-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{{language}}.{part}", part=parts)
    resources:
        time="08:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_cf_{language}_{variant}_{num_toks}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_DATA_DIR}/{{wildcards.language}}/{{wildcards.variant}}
        cd counterfactual
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}.train.conllu > ../{CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}.valid.conllu > ../{CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid
        python apply_counterfactual_grammar.py --language {{wildcards.language}} --model {{wildcards.variant}} --filename ../{PARSE_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}.test.conllu > ../{CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test
        """.format(CF_DATA_DIR="data/wiki40b-txt-cf-diff-sizes", PARSE_DIR="data/wiki40b-txt-parsed-diff-sizes")

rule apply_bpe_diff_sizes:
    input:
        expand("data/wiki40b-txt-cf-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
        "data/bpe_codes_cf/30k/{language}.codes"
    output:
        expand("data/wiki40b-txt-cf-bpe-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    resources:
        time="01:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_apply_bpe_{language}_{variant}_{num_toks}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        {FASTBPE_PATH} applybpe {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {CF_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {FASTBPE_OUTPATH}/{{wildcards.language}}.codes
        """.format(CF_BPE_DATA_DIR="data/wiki40b-txt-cf-bpe-diff-sizes", FASTBPE_OUTPATH=FASTBPE_OUTPATH, FASTBPE_PATH=FASTBPE_PATH, CF_DATA_DIR="data/wiki40b-txt-cf-diff-sizes")

# binarize for fairseq training
rule prepare_fairseq_data_diff_sizes:
    input:
        expand("data/wiki40b-txt-cf-bpe-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{{language}}.{part}", part=parts),
    output:
        expand("data/data-bin-cf-bpe-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{part}.bin", part=parts),
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_preprocess_{language}_{variant}_{num_toks}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        rm -r {PREPROCESSED_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}
        mkdir -p {PREPROCESSED_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}
        fairseq-preprocess \
            --only-source \
            --trainpref {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.train \
            --validpref {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.valid \
            --testpref {CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test \
            --destdir {PREPROCESSED_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}} \
            --bpe fastbpe \
            --workers 20 
        """.format(CF_BPE_DATA_DIR="data/wiki40b-txt-cf-bpe-diff-sizes", PREPROCESSED_DATA_DIR="data/data-bin-cf-bpe-diff-sizes")

# train the models
rule train_language_models_diff_sizes:
    input:
        "data/data-bin-cf-bpe-diff-sizes/{num_toks}/{language}/{variant}/train.bin",
        "data/data-bin-cf-bpe-diff-sizes/{num_toks}/{language}/{variant}/valid.bin",
    output:
        "data/checkpoint-cf-bpe-diff-sizes/{num_toks}/{model_seed}/{language}/{variant}/checkpoint_best.pt"
    resources:
        time="24:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_train_{language}_{variant}_{num_toks}_{model_seed}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {CHECKPOINT_DIR}/{{wildcards.num_toks}}/{{wildcards.model_seed}}/{{wildcards.language}}/{{wildcards.variant}}
        cd data
        bash train_model_transformer.sh ../{PREPROCESSED_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}} \
            ../{CHECKPOINT_DIR}/{{wildcards.num_toks}}/{{wildcards.model_seed}}/{{wildcards.language}}/{{wildcards.variant}}
        """.format(PREPROCESSED_DATA_DIR="data/data-bin-cf-bpe-diff-sizes", CHECKPOINT_DIR="data/checkpoint-cf-bpe-diff-sizes")

rule eval_language_models_diff_sizes:
    input:
        "data/checkpoint-cf-bpe-diff-sizes/{num_toks}/{model_seed}/{language}/{variant}/checkpoint_best.pt",
        "data/wiki40b-txt-cf-bpe-diff-sizes/{num_toks}/{language}/{variant}/{language}.test",
        "data/data-bin-cf-bpe-diff-sizes/{num_toks}/{language}/{variant}/test.bin"
    output:
        "evaluation/perps-cf-diff-sizes/{num_toks}/{model_seed}/{language}-{variant}.pt"
    resources:
        time="4:00",
        num_cpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
    log:
        "data/logs_thc/log_eval_{language}_{variant}_{num_toks}_{model_seed}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p {EVAL_RESULTS_DIR}/{{wildcards.num_toks}}/{{wildcards.model_seed}}
        cd data
        python per_example_perp.py {BASE_DIR}/{CHECKPOINT_DIR}/{{wildcards.num_toks}}/{{wildcards.model_seed}}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{PREPROCESSED_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}} {BASE_DIR}/{CF_BPE_DATA_DIR}/{{wildcards.num_toks}}/{{wildcards.language}}/{{wildcards.variant}}/{{wildcards.language}}.test {BASE_DIR}/{EVAL_RESULTS_DIR}/{{wildcards.num_toks}}/{{wildcards.model_seed}}/{{wildcards.language}}-{{wildcards.variant}}.pt
        """.format(BASE_DIR=BASE_DIR, CHECKPOINT_DIR="data/checkpoint-cf-bpe-diff-sizes", PREPROCESSED_DATA_DIR="data/data-bin-cf-bpe-diff-sizes", CF_BPE_DATA_DIR="data/wiki40b-txt-cf-bpe-diff-sizes", EVAL_RESULTS_DIR="evaluation/perps-cf-diff-sizes")

rule postprocess_eval_output_diff_sizes:
    input:
        expand("evaluation/perps-cf-diff-sizes/{num_toks}/{model_seed}/{language}-{variant}.pt", language=["en", "ru", "hu", "tr", "hi"], variant=variants, num_toks=[2222222, 6666666], model_seed=[1,2])
    output:
        "evaluation/eval_results_cf_diff_sizes.feather"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=32000,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_postprocess_eval_output_diff_sizes.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python evaluation.py --make_csv --perps_file_pattern 'perps-cf-diff-sizes/*/*/*.pt' --out_file 'eval_results_cf_diff_sizes.feather'
        """

rule make_plotting_inputs_diff_sizes:
    input:
        "evaluation/eval_results_cf_diff_sizes.feather"
    output:
        "evaluation/plot_csv_diff_sizes/surprisal_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/surprisal_variance_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/doc_initial_var.csv",
        "evaluation/plot_csv_diff_sizes/surprisal_deviations_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/delta_surps_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/infs_1.1_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/infs_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/avg_surps_plot_vals.csv",
        "evaluation/plot_csv_diff_sizes/delta_surps_by_tok.csv",
        "evaluation/plot_csv_diff_sizes/max_surps_plot_vals.csv",
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
    shell:
        """
        cd evaluation
        mkdir -p plot_csv_diff_sizes
        python make_plotting_inputs.py --inputfile eval_results_cf_diff_sizes.feather --data_dir plot_csv_diff_sizes
        """