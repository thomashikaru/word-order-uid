import streamlit as st
from spacy_streamlit import visualize_parser
import corpus_iterator
import corpus_iterator_funchead
from apply_counterfactual_grammar import orderSentence, reorder_heads, get_dl
import random
from math import ceil

# set wide layout
st.set_page_config(layout="wide")

deps = [
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "lifted_case",
    "lifted_cc",
    "lifted_cop",
    "lifted_mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "vocative",
    "xcomp",
]

N_DEPS = len(deps)
N_COLS = 9

example = """# text = The quick brown fox jumped over the lazy dog
1	The	the	DET	DEF	Definite=Def|PronType=Art	4	det	_	TokenRange=0:3
2	quick	quick	ADJ	POS	Degree=Pos	4	amod	_	TokenRange=4:9
3	brown	brown	ADJ	POS	Degree=Pos	4	amod	_	TokenRange=10:15
4	fox	fox	NOUN	SG-NOM	Number=Sing	5	nsubj	_	TokenRange=16:19
5	jumped	jump	VERB	PAST	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	_	TokenRange=20:26
6	over	over	ADP	_	_	9	case	_	TokenRange=27:31
7	the	the	DET	DEF	Definite=Def|PronType=Art	9	det	_	TokenRange=32:35
8	lazy	lazy	ADJ	POS	Degree=Pos	9	amod	_	TokenRange=36:40
9	dog	dog	NOUN	SG-NOM	Number=Sing	5	obl	_	TokenRange=41:44"""


def verify_input(text):
    lines = text.split("\n")
    lines = filter(lambda x: not x.startswith("#"), lines)
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 10:
            return False
        if int(fields[0]) != i + 1:
            return False
    return True


# convert sentence from Michael's format to displaCy render (manual) format
def convert_sentence(sentence):
    words, arcs = [], []
    for word in sentence:
        words.append({"text": word["word"], "tag": word["posUni"]})
        idx, head = int(word["index"]), int(word["reordered_head"])
        if head == 0:
            continue
        dir = "right"
        if idx > head:
            idx, head = head, idx
            dir = "left"
        arcs.append(
            {"start": idx - 1, "end": head - 1, "label": word["dep"], "dir": dir,}
        )
    return {"words": words, "arcs": arcs}


# initialize random weights for dh and distance weights
@st.cache(allow_output_mutation=True)
def initialize_weights():
    dh_weights, distance_weights = {}, {}
    for x in deps:
        dh_weights[x] = random.random() - 0.5
        distance_weights[x] = random.random() - 0.5
    return dh_weights, distance_weights


dh_weights, distance_weights = initialize_weights()

# page title
st.title("Counterfactual Grammar Visualization")

# input original parse (includes default example)
st.header("Original Parse")
text = st.text_area("Enter text in CoNLL format:", value=example, height=400)
text = text.strip()

if not verify_input(text):
    st.error(
        "The input text is not formatted correctly. A default text will be used instead."
    )
    text = example


# get the sentence
corpus = corpus_iterator.CorpusIterator("", "English")
sentence, newdoc = corpus.processSentence(text)

try:
    sentence = corpus_iterator_funchead.reverse_content_head(sentence)
except IndexError:
    st.error(
        "Something went wrong (probably a problem with the input formatting). Using a default sentence instead."
    )
    sentence, newdoc = corpus.processSentence(example)
    sentence = corpus_iterator_funchead.reverse_content_head(sentence)

# where to display parse tree (create it but populate it later, after sliders)
treebox = st.container()

dlbox = st.container()

# sliders for dh weights
st.header("Dependent-Head Weights")
st.caption(
    "A positive weight for a given relation means that the dependent will occur before the head in linear order."
)
dhcols = st.columns(N_COLS)
# n_rows = ceil(N_DEPS / N_COLS)
start = 0
for i, dhcol in enumerate(dhcols):
    with dhcol:
        slider_vals = {}
        # for dep in deps[start : min(start + n_rows, N_DEPS)]:
        for dep in deps[start::N_COLS]:
            if dep in dh_weights:
                dh_weights[dep] = st.slider(
                    dep, -1.0, 1.0, dh_weights[dep], key="dh" + dep, format="%.2f"
                )
        start += 1

# sliders for distance weights
st.header("Distance Weights")
st.caption(
    "For dependents on the same side of a head, those with higher weights will be placed farther from the head in linear order."
)
distcols = st.columns(N_COLS)
start = 0
for i, distcol in enumerate(distcols):
    with distcol:
        slider_vals = {}
        # for dep in deps[start : min(start + n_rows, N_DEPS)]:
        for dep in deps[start::N_COLS]:
            if dep in distance_weights:
                distance_weights[dep] = st.slider(
                    dep,
                    -1.0,
                    1.0,
                    distance_weights[dep],
                    key="dist" + dep,
                    format="%.2f",
                )
        start += 1

st.header("Grammar Type")
grammar = st.selectbox("Select one of the following", ("RANDOM", "MIN_DL_PROJ"))

st.header("About")
st.caption(
    "This visualizer is built on top of code from the following GitHub repo: https://github.com/m-hahn/grammar-optim, which accompanies the paper 'Universals of word order reflect optimization of grammars for efficient communication' by Hahn et al. (2020)."
)

# update the treebox
with treebox:
    try:
        sentence = orderSentence(sentence, grammar, dh_weights, distance_weights)
        reorder_heads(sentence)
        for i, s in enumerate(sentence):
            s["index"] = i + 1
        data = convert_sentence(sentence)
        dl = get_dl(sentence)
        visualize_parser(
            data,
            manual=True,
            displacy_options={
                "distance": 150,
                "word_spacing": 45,
                "offset_x": 45,
                "bg": "#FFFFFF",
            },
        )
    except IndexError:
        st.error(
            "Something went wrong (probably a problem with the input formatting). Please try reloading the page."
        )

with dlbox:
    st.header("Total Dependency Length")
    st.subheader(f"{dl}")

