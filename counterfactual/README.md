# Counterfactual

This directory deals with generating and visualizing counterfactual grammars for languages. 

## File Overview

- `apply_counterfactual_grammar.py`: contains functions for applying a counterfactual grammar to a language dataset. This could include:
    - Real ordering (useful because it applies the same preprocessing that it does to counterfactual orders for a fair comparison)
    - Random orderings (dependent-head directionality weights and distance weights)
    - Specific orderings (references grammar specifications contained in `grammars/auto-summary-lstm.tsv`)
- `corpus_iterator.py`: iterator for loading and processing a language dataset in CoNLL-U (Universal Dependencies) format. 
- `corpus_iterator_funchead.py`: a wrapper around `corpus_iterator` that applies a convention change in how head/dependent is determined for certain syntactic relationships, such as case, cc, etc. 
- `dep_parse.py`: uses UDPipe to generate dependency parses in CoNLL-U format for a plain text language dataset. 
- `vis.py`: contains Streamlit app for visualizing the effects of tweaking parameters in a counterfactual grammar.
- `failure_cases.txt` and `failure_cases.conllu`: contains cases where `apply_counterfactual_grammar` failed due to problems with applying the funchead reversal in `corpus_iterator_funchead`. It turns out that this occured when a dependency relation was a child of the same dependency relation. 