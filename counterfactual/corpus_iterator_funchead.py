from corpus_iterator import CorpusIterator

header = [
    "index",
    "word",
    "lemma",
    "posUni",
    "posFine",
    "morph",
    "head",
    "dep",
    "_",
    "_",
]


def reverse_content_head(sentence):
    """Apply dependency parse convention change (Deviation from vanilla UD)

    Args:
        sentence (List[Dict[str,int]]): a list of dictionaries, each corresponding to a word,
        with the UD header names as dictionary keys

    Returns:
        List[Dict[str,int]]: same format as input
    """
    CH_CONVERSION_ORDER = ["cc", "case", "cop", "mark"]
    # find paths that should be reverted
    for dep in CH_CONVERSION_ORDER:
        for i in range(len(sentence)):
            if sentence[i]["dep"] == dep or sentence[i]["dep"].startswith(dep + ":"):
                head = sentence[i]["head"] - 1
                grandp = sentence[head]["head"] - 1
                assert head > -1

                # grandp -> head -> i
                # grandp -> i -> head
                sentence[i]["head"] = grandp + 1
                sentence[head]["head"] = i + 1

                sentence[i]["dep"] = sentence[head]["dep"]
                sentence[head]["dep"] = "lifted_" + dep
                assert sentence[i]["index"] == i + 1
    return sentence


class CorpusIteratorFuncHead:
    def __init__(self, filename, language, partition="train", storeMorph=False):
        self.basis = CorpusIterator(
            filename, language, partition=partition, storeMorph=storeMorph,
        )

    def permute(self):
        self.basis.permute()

    def length(self):
        return self.basis.length()

    def iterator(self, rejectShortSentences=False):
        iterator = self.basis.iterator(rejectShortSentences=rejectShortSentences)
        for sentence, newdoc in iterator:
            reverse_content_head(sentence)
            yield sentence, newdoc

    def getSentence(self, index):
        sentence, newdoc = self.basis.getSentence(index)
        return reverse_content_head(sentence), newdoc

