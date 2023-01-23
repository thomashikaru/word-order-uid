
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
        "evaluation/plots/joint_surprisal_and_variance.png"


######################################
### wiki40b - 20m tokens
######################################

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
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {BASE_DIR}/{RAW_DATA_DIR} --output_prefix {SAMPLED_DATA_DIR} --wiki40b
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

rule all_cf_data_wiki40b_20m:
    input:
        expand("data/wiki40b-txt-cf/{language}/{variant}/{language}.{part}", language=languages, variant=variants, part=parts),

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

rule all_dataset_prep_wiki40b:
    input:
        expand("data/data-bin-cf-bpe/{language}/{variant}/{part}.bin", language=languages, variant=variants, part=parts)

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

rule all_models_train_wiki40b:
    input: 
        expand("data/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt", language=languages, variant=variants)

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

rule postprocess_wiki40b:
    input:
        "evaluation/perps-cf/{language}-{variant}.pt"
    output:
        "evaluation/perps-cf/{language}-{variant}.csv"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=16000,ngpus_excl_p=0]",
        mem_per_cpu=16000,
    log:
        "data/logs_thc/log_postprocess_wiki40b_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python postprocess_eval_results.py --inputfile perps-cf/{wildcards.language}-{wildcards.variant}.pt --language {wildcards.language} --variant {wildcards.variant} --num_toks 20000000 --model_seed none --dataset wiki40b
        """

rule postprocess_wiki40b_all:
    input:
        expand("evaluation/perps-cf/{language}-{variant}.csv", language=languages, variant=variants)
    output:
        "evaluation/perps-cf/results_summary.csv"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
        mem_per_cpu=8000,
    log:
        "data/logs_thc/log_postprocess_wiki40b_all.out"
    run:
        import glob
        import pandas as pd
        filenames = glob.glob("evaluation/perps-cf/*.csv")
        dfs = [pd.read_csv(f) for f in filenames]
        df = pd.concat(dfs)
        df.to_csv("evaluation/perps-cf/results_summary.csv", index=False)


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

######################################
### cc100
######################################

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

rule all_cf_data_cc100:
    input:
        expand("data/cc100/txt-cf/{language}/{variant}/{language}.{part}", language=languages_cc100, variant=variants, part=parts),

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

rule all_dataset_prep_cc100:
    input:
        expand("data/cc100/data-bin-cf-bpe/{language}/{variant}/{part}.bin", language=languages_cc100, variant=variants, part=parts)

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
rule all_models_train_cc100:
    input: 
        expand("data/cc100/checkpoint-cf-bpe/{language}/{variant}/checkpoint_best.pt", language=languages_cc100, variant=variants)

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

rule postprocess_cc100:
    input:
        "evaluation/cc100/perps-cf/{language}-{variant}.pt"
    output:
        "evaluation/cc100/perps-cf/{language}-{variant}.csv"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=16000,ngpus_excl_p=0]",
        mem_per_cpu=16000,
    log:
        "data/logs_thc/log_postprocess_cc100_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python postprocess_eval_results.py --inputfile cc100/perps-cf/{wildcards.language}-{wildcards.variant}.pt --language {wildcards.language} --variant {wildcards.variant} --num_toks None --model_seed None --dataset cc100
        """

rule postprocess_cc100_all:
    input:
        expand("evaluation/cc100/perps-cf/{language}-{variant}.csv", language=languages_cc100, variant=variants)
    output:
        "evaluation/cc100/perps-cf/results_summary.csv"
    resources:
        time="4:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
        mem_per_cpu=8000,
    log:
        "data/logs_thc/log_postprocess_cc100_all.out"
    run:
        import glob
        import pandas as pd
        filenames = glob.glob("evaluation/cc100/perps-cf/*.csv")
        dfs = [pd.read_csv(f) for f in filenames]
        df = pd.concat(dfs)
        df.to_csv("evaluation/cc100/perps-cf/results_summary.csv", index=False)


#######################################################
### wiki40b - different sizes (1/3 and 1/9) of original
#######################################################

# sample wiki40b datasets
rule sample_wiki40b_data_diff_sizes:
    input:
        expand("data/raw_data/wiki40b-txt/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-sampled-diff-sizes/{{num_toks}}/{{language}}.{part}", part=parts)
    resources:
        time="12:00",
        time_slurm="12:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
        mem_per_cpu=4000,
        mem_per_gpu=0,
    log:
        "data/logs_thc/log_sample_{language}_{num_toks}.out"
    shell: 
        f"""
        cd data
        mkdir -p wiki40b-txt-sampled-diff-sizes/{{wildcards.num_toks}}
        python sample.py --lang_code_list {{wildcards.language}} --input_prefix {BASE_DIR}/{RAW_DATA_DIR} --output_prefix wiki40b-txt-sampled-diff-sizes/{{wildcards.num_toks}} --num_train_tokens {{wildcards.num_toks}} --wiki40b
        """

rule do_dependency_parsing_diff_sizes:
    input:
        expand("data/wiki40b-txt-sampled-diff-sizes/{{num_toks}}/{{language}}.{part}", part=parts)
    output:
        expand("data/wiki40b-txt-parsed-diff-sizes/{{num_toks}}/{{language}}.{part}.conllu", part=parts)
    resources:
        time="24:00",
        time_slurm="24:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=2048,ngpus_excl_p=0]",
        mem_per_cpu=2048,
        mem_per_gpu=0,
    log:
        "data/logs_thc/log_parse_{language}_{num_toks}.out"
    shell:
        f"""
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p data/wiki40b-txt-parsed-diff-sizes/{{wildcards.num_toks}}
        cd counterfactual
        python dep_parse.py --lang {{wildcards.language}} --data_dir {BASE_DIR}/data/wiki40b-txt-sampled-diff-sizes/{{wildcards.num_toks}} --parse_dir ../data/wiki40b-txt-parsed-diff-sizes/{{wildcards.num_toks}} --partitions 'train,test,valid'
        """

# make counterfactual datsets for each language
rule make_cf_data_diff_sizes:
    input:
        expand("data/wiki40b-txt-parsed-diff-sizes/{{num_toks}}/{{language}}.{part}.conllu", part=parts),
        "counterfactual/freqs/{language}.csv"
    output:
        expand("data/wiki40b-txt-cf-diff-sizes/{{num_toks}}/{{language}}/{{variant}}/{{language}}.{part}", part=parts)
    resources:
        time="08:00",
        time_slurm="08:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
        mem_per_cpu=4096,
        mem_per_gpu=0,
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
        time_slurm="01:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=4000,ngpus_excl_p=0]",
        mem_per_cpu=4000,
        mem_per_gpu=0,
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
        time_slurm="04:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
        mem_per_cpu=8000,
        mem_per_gpu=0,
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
        time_slurm="24:00:00",
        num_cpus=1,
        num_gpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
        mem_per_cpu=1000,
        mem_per_gpu=30000,
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
        time_slurm="04:00:00",
        num_cpus=1,
        num_gpus=1,
        select="select[gpu_mtotal0>=10000]",
        rusage="rusage[mem=30000,ngpus_excl_p=1]",
        mem_per_cpu=1000,
        mem_per_gpu=30000,
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

rule postprocess_diff_sizes:
    input:
        "evaluation/perps-cf-diff-sizes/{num_toks}/{model_seed}/{language}-{variant}.pt"
    output:
        "evaluation/perps-cf-diff-sizes/{num_toks}/{model_seed}/{language}-{variant}.csv"
    resources:
        time="4:00",
        time_slurm="04:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=16000,ngpus_excl_p=0]",
        mem_per_cpu=16000,
        mem_per_gpu=0,
    log:
        "data/logs_thc/log_postprocess_diff_sizes_{num_toks}_{model_seed}_{language}_{variant}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        cd evaluation
        python postprocess_eval_results.py --inputfile perps-cf-diff-sizes/{wildcards.num_toks}/{wildcards.model_seed}/{wildcards.language}-{wildcards.variant}.pt --language {wildcards.language} --variant {wildcards.variant} --num_toks {wildcards.num_toks} --model_seed {wildcards.model_seed} --dataset wiki40b
        """

rule postprocess_diff_sizes_all:
    input:
        expand("evaluation/perps-cf-diff-sizes/{num_toks}/{model_seed}/{language}-{variant}.csv", language=languages, variant=variants, num_toks=[2222222, 6666666, 20000000], model_seed=[1,2])
    output:
        "evaluation/perps-cf-diff-sizes/results_summary.csv"
    resources:
        time="4:00",
        time_slurm="04:00:00",
        num_cpus=1,
        num_gpus=0,
        select="",
        rusage="rusage[mem=8000,ngpus_excl_p=0]",
        mem_per_cpu=8000,
        mem_per_gpu=0,
    log:
        "data/logs_thc/log_postprocess_diff_sizes_all.out"
    run:
        import glob
        import pandas as pd
        filenames = glob.glob("evaluation/perps-cf-diff-sizes/*/*/*.csv")
        dfs = [pd.read_csv(f) for f in filenames]
        df = pd.concat(dfs)
        df.to_csv("evaluation/perps-cf-diff-sizes/results_summary.csv", index=False)


######################################
### dependency length measurement
######################################

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

######################################
### entropy measurement
######################################

rule measure_entropy:
    input:
        expand("data/wiki40b-txt-parsed/{{language}}.test.conllu"),
    output:
        expand("counterfactual/wiki40b-entropy/{{language}}.csv")
    resources:
        time="04:00",
        num_cpus=1,
        select="",
        rusage="rusage[mem=4096,ngpus_excl_p=0]",
    log:
        "data/logs_thc/log_measure_entropy_{language}.out"
    shell:
        """
        module load gcc/6.3.0
        module load python_gpu/3.8.5 hdf5 eth_proxy
        module load geos libspatialindex
        mkdir -p wiki40b-entropy
        cd counterfactual
        python sov_entropy.py --language {{wildcards.language}} --filename ../{PARSE_DIR}/{{wildcards.language}}.test.conllu --outfile wiki40b-entropy/{{wildcards.language}}.csv
        """.format(PARSE_DIR=PARSE_DIR)

rule measure_entropy_wiki40b:
    input:
        expand("counterfactual/wiki40b-entropy/{{language}}.csv", language=languages)
    output:
        "counterfactual/wiki40b-entropy/summary.csv"
    log:
        "data/logs_thc/log_measure_entropy_summary.out"
    run:
        import glob
        import pandas as pd
        filenames = glob.glob("counterfactual/wiki40b-entropy/*.csv")
        dfs = [pd.read_csv(f) for f in filenames]
        df = pd.concat(dfs)
        df.to_csv("counterfactual/wiki40b-entropy/summary.csv", index=False)