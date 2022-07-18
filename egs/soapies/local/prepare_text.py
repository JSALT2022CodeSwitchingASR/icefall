#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# 	       2022  Xiaomi Crop.        (authors: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from pathlib import Path
import unicodedata

from lhotse import SupervisionSet

lang_shortname = {
    "xhosa": "xh",
    "zulu": "zu",
    "sesotho": "st",
    "setswana": "tn",
}

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def add_line(words, line):
    for word in line.split():
        words.add(word)

def preprocess_text(text):
    return remove_accents(text.strip().upper().replace("-", " ").replace(".", "").replace("_", ""))

def prepare_text(lang):
    src_dir = Path("data/manifests") / lang
    texts_dir = Path("data/texts")
    output_dir = texts_dir / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_parts = (
        "train",
        "dev",
        "test",
    )

    prefix = f"soapies-{lang}"
    suffix = "jsonl.gz"
    supervisions = {}
    for split in dataset_parts:
        supervisions[split] = SupervisionSet.from_jsonl(
            src_dir / f"{prefix}_supervisions_{split}.{suffix}"
        )

    words = set()

    logging.info("creating LM train corpus")
    with open(output_dir / "text_train_corpus.txt", "w") as f:
        for split in ("train", "dev"):
            for supervision in supervisions[split]:
                text = preprocess_text(supervision.text)
                add_line(words, text)
                print(text, file=f)

        with open(texts_dir / f"bible_{lang}.txt", "r") as src:
            for line in src:
                line = preprocess_text(line)
                add_line(words, line)
                print(line, file=f)

        with open(texts_dir / f"nchlt_{lang_shortname[lang]}.txt", "r") as src:
            for line in src:
                line = preprocess_text(line)
                add_line(words, line)
                print(line, file=f)

        if lang == "xhosa" or lang == "zulu":
            with open(texts_dir / f"{lang}.txt", "r") as src:
                for line in src:
                    line = line.strip().upper()
                    add_line(words, line)
                    print(line, file=f)

    with open(output_dir / "words.txt", "w") as f:
        for word in sorted(words):
            if word == "<UNK>":
                continue
            print(word, file=f)

    #with open(output_dir / "text_test.txt", "w") as f:
    #    for supervision in supervisions["test"]:
    #            print(supervision.text, file=f)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lang",
        help="language for which to prepare the text")
    args = parser.parse_args()

    prepare_text(args.lang)

