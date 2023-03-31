# Word Order UID - TACL 

This repo contains code accompanying the TACL paper "A Cross-Linguistic Pressure for Uniform Information Density in Word Order". 

Corresponding author: Thomas Hikaru Clark (thclark at mit dot edu)

## Snakemake workflow

The workflow is managed using [Snakemake](https://snakemake.readthedocs.io/en/stable/).

Commands to run:
LSF:
```bash
snakemake {rule} --cores {cores} --cluster "sbatch --time={resources.time} -n {resources.num_cpus} --mem-per-cpu={resources.mem_per_cpu} --gpus={resources.num_gpus} --gres=gpumem:{resources.mem_per_gpu} -o {log}"
```
Slurm:
```bash
snakemake {rule} --cores {cores} --slurm
```

## Data

The data for our experiments come from two sources: [Wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) and [CC100](https://huggingface.co/datasets/cc100). 

## Counterfactual Grammars

We use the counterfactual grammar formalism of [Hahn et al. (2020)](https://www.pnas.org/doi/10.1073/pnas.1910923117). In this formalism, weights are assigned to each dependency relation type in the Universal Dependencies paradigm. These weights are then used to linearize the hierarchical structure of a sentence's dependency parse. An interactive visualizer can be found [here](https://share.streamlit.io/thomashikaru/dependency-tweak/vis.py). 

## Training Models

### Hyperparameters

Model training hyperparameters are specified in the file `data/train_model_transformer.sh`:

```bash
fairseq-train --task language_modeling \
	$DATA_DIR \
	--save-dir $SAVE_DIR \
	--arch transformer_lm \
    --share-decoder-input-output-embed \
	--dropout 0.1 \
	--optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
	--lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
	--tokens-per-sample 512 \
    --sample-break-mode none \
	--max-tokens 512 \
    --update-freq 64 \
	--fp16 \
	--max-update 50000 \
    --max-epoch 35 \
    --patience 3  \
	--seed $RANDOM_SEED \
	--keep-last-epochs 5
```

## Evaluation

## Plots

## Statistical Modeling