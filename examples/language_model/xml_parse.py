import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict
import glob
from iso_639 import lang_codes
import re

# remove <quote> and <report> tags which interfere with getting text
def clean(s):
    s = re.sub("(<quote[^>]*>)|(</quote>)|(<report[^>]*>)|(</report>)", "", s)
    s = re.sub("\s+", " ", s)
    return s


# parse a single XML file and update result dict with wordcounts
def parse_xml_wordcount(input_file, result):
    with open(input_file) as f:
        xml = clean(f.read())
    session = ET.fromstring(xml)

    for chapter in session:
        for turn in chapter:
            if turn.tag != "turn":
                continue
            speaker = turn[0]
            for text in speaker:
                language = text.attrib["language"]
                for p in text:
                    if p.text == None or p.text.strip() == "":
                        continue
                    result[language] += len(p.text.split())
    return result


# parse a single XML file and append data to per-language .txt files
def parse_xml_and_save(input_file, output_prefix):
    with open(input_file) as f:
        xml = clean(f.read())
    session = ET.fromstring(xml)
    result = defaultdict(list)

    for chapter in session:
        for turn in chapter:
            if turn.tag != "turn":
                continue
            speaker = turn[0]
            for text in speaker:
                language = text.attrib["language"]
                sentences = []
                for p in text:
                    if p.text == None or p.text.strip() == "":
                        continue
                    sentences.append(p.text)
                if len(sentences) == 0:
                    continue
                result[language].append(" ".join(sentences))

    for lang, data in result.items():
        output_file = output_prefix + lang + ".txt"
        with open(output_file, "a") as f:
            f.write("\n".join(data) + "\n")


# get wordcount by language for Europarl XML files
def get_lang_counts(args):
    result = defaultdict(int)

    for input_file in glob.glob(args.input):
        result = parse_xml_wordcount(input_file, result)

    print(f"{len(result.keys())} languages represented")
    for lang, data in result.items():
        n = data
        print(f"{lang_codes[lang] : <20}: {n : >15} tokens")


# parse Europarl XML corpus and save .txt files for each language
def generate_dataset(args):
    for input_file in glob.glob(args.input):
        parse_xml_and_save(input_file, args.output_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="glob pattern specifying file paths to input XML files"
    )
    parser.add_argument(
        "--output_prefix",
        help="path and prefix to desired output location for .txt files",
    )
    parser.add_argument(
        "--wordcount",
        action="store_true",
        help="use this flag to output wordcounts only",
    )
    args = parser.parse_args()

    if args.wordcount:
        get_lang_counts(args)
    else:
        generate_dataset(args)
