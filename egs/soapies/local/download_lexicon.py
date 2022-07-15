"""Prepare the `lang` directory for the specified language."""

import argparse
import gzip
import logging
from pathlib import Path
import urllib.request

# Use the same logging format as in lhotse.
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO
)

# List of the languages supported.
languages = ['sesotho', 'setswana', 'xhosa', 'zulu']

# Google drive file identifier.
fileids = {
    'sesotho': '1tAhG6ZpFvFLCyqUQuGkRH5YVARwUeHUh',
    'setswana': '1cciROdDzXm2osXU2-TuvEJFrg197R2YM',
    'xhosa': '1lVweSHXrmhDZspSKrZXM48UnybY5l7od',
    'zulu': '1sU39l5i_TzFog_EJQh4h0La_Q86yBCSQ',
}

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lexicon_tmp = outdir / "lexicon.tmp"
    lexicon = outdir / "lexicon.txt"
    if not lexicon.is_file():
        fid = fileids[args.lang]
        url = f"https://drive.google.com/uc?export=download&id={fid}"
        logging.info(f"downloading lexicon from {url}")

        response = urllib.request.urlopen(url)
        with open(lexicon_tmp, "wb") as f:
            f.write(gzip.decompress(response.read()))

        toremove = set(["<s>", "</s>", "SIL-ENCE", "[s]"])
        with open(lexicon_tmp, "r") as f_in, open(lexicon, "w") as f_out:
            print("<SIL> <SIL>", file=f_out)
            print("<UNK> <UNK>", file=f_out)
            for line in f_in:
                tokens = line.strip().split()
                if tokens[0] not in toremove:
                    print(tokens[0].upper(), " ".join(tokens[1:]), file=f_out)
        lexicon_tmp.unlink()
    else:
        logging.info(f'lexicon already extracted to {lexicon}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lang', choices=languages, help='language to prepare')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    main(args)
