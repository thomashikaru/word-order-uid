import os
import random
import sys

HEADER = [
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


def read_conllu_file(path):
    with open(path) as f:
        data = f.read().strip()
    data = data.split("\n\n")
    return data


def readUDCorpus(language, partition):
    # basePaths = ["/u/scr/mhahn/grammar-optim_ADDITIONAL/corpora/"]
    basePaths = ["/Users/thomasclark/datasets/ud-treebanks-v2.1"]
    files = []
    while len(files) == 0:
        if len(basePaths) == 0:
            print("No files found")
            raise IOError
        basePath = basePaths[0]
        del basePaths[0]
        files = os.listdir(basePath)
        files = list(filter(lambda x: x.startswith("UD_" + language), files))
    data = []
    for name in files:
        if "Sign" in name:
            print("Skipping " + name)
            continue
        assert "Sign" not in name
        if "Chinese-CFL" in name:
            print("Skipping " + name)
            continue
        suffix = name[len("UD_" + language) :]
        subDirectory = basePath + "/" + name
        subDirFiles = os.listdir(subDirectory)
        partitionHere = partition

        candidates = list(
            filter(
                lambda x: "-ud-" + partitionHere + "." in x and x.endswith(".conllu"),
                subDirFiles,
            )
        )
        if len(candidates) == 0:
            print("Did not find " + partitionHere + " file in " + subDirectory)
            continue
        if len(candidates) == 2:
            candidates = list(filter(lambda x: "merged" in x, candidates))
        assert len(candidates) == 1, candidates
        try:
            dataPath = subDirectory + "/" + candidates[0]
            with open(dataPath, "r") as inFile:
                newData = inFile.read().strip().split("\n\n")
                assert len(newData) > 1
                data = data + newData
        except IOError:
            print("Did not find " + dataPath)

    assert len(data) > 0, (language, partition, files)

    print(
        f"Read {len(data)} sentences from {len(files)} {partition} datasets.",
        file=sys.stderr,
    )
    return data


class CorpusIterator:
    def __init__(
        self, filename, language, partition="train", storeMorph=False,
    ):

        # store args
        self.storeMorph = storeMorph

        # read corpus: data = list of parse strings
        # data = readUDCorpus(language, partition)
        data = read_conllu_file(filename)

        self.data = data
        self.filename = filename
        self.partition = partition
        self.language = language
        assert len(data) > 0, (filename, language, partition)

    def permute(self):
        random.shuffle(self.data)

    def length(self):
        return len(self.data)

    def processSentence(self, sentence):

        # split each parse into lines (one line per word)
        # split each line into fields (separated by tabs)
        # sentence = list of lists of fields
        sentence = list(map(lambda x: x.split("\t"), sentence.split("\n")))

        result = []
        for i in range(len(sentence)):
            if sentence[i][0].startswith("#"):
                continue
            if "-" in sentence[i][0]:  # if it is NUM-NUM
                continue
            if "." in sentence[i][0]:
                continue

            # sentence = list of dicts, where each key is a field name (see HEADER)
            sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(HEADER)])
            sentence[i]["head"] = int(sentence[i]["head"])
            sentence[i]["index"] = int(sentence[i]["index"])
            sentence[i]["word"] = sentence[i]["word"].lower()

            if self.storeMorph:
                sentence[i]["morph"] = sentence[i]["morph"].split("|")

            sentence[i]["dep"] = sentence[i]["dep"].lower()
            if self.language == "LDC2012T05" and sentence[i]["dep"] == "hed":
                sentence[i]["dep"] = "root"
            if self.language == "LDC2012T05" and sentence[i]["dep"] == "wp":
                sentence[i]["dep"] = "punct"

            result.append(sentence[i])
        return result

    def getSentence(self, index):
        result = self.processSentence(self.data[index])
        return result

    def iterator(self, rejectShortSentences=False):
        for sentence in self.data:
            if len(sentence) < 3 and rejectShortSentences:
                continue
            yield self.processSentence(sentence)

