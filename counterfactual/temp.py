import corpus_iterator, corpus_iterator_funchead
import json
import sys
import ast

if __name__ == "__main__":

    with open("failure_cases.txt") as f:
        examples = f.readlines()

    for example in examples:
        lines = []
        data = ast.literal_eval(example)
        for word in data:
            lines.append("\t".join([str(x) for x in word.values()]))
            lines[-1] += "\t_"
        text = "\n".join(lines).strip()
        sys.stdout.write(text)
        sys.stdout.write("\n\n")
    # corpus = corpus_iterator.CorpusIterator("", "English")
    # sentence, newdoc = corpus.processSentence(text)
    # sentence = corpus_iterator_funchead.reverse_content_head(sentence)
    # for word in sentence:
    #     print("\t".join([str(x) for x in list(word.values())]))
