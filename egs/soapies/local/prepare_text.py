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
"""
Compute fbank features for a language of the soapies corpus.

It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

from lhotse import SupervisionSet

def prepare_text(lang):
    src_dir = Path("data/manifests") / lang
    output_dir = Path("data/texts") / lang
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

    with open(output_dir / "text_train+dev.txt", "w") as f:
        for split in ("train", "dev"):
            for supervision in supervisions[split]:
                print(supervision.text, file=f)

    with open(output_dir / "text_test.txt", "w") as f:
        for supervision in supervisions["test"]:
                print(supervision.text, file=f)


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

