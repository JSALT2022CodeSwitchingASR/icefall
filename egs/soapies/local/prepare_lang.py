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

    lexicon = outdir / "lexicon"
    if not lexicon.is_file():
        fid = fileids[args.lang]
        response = urllib.request.urlopen(f"https://drive.google.com/uc?export=download&id={fid}")
        with open(lexicon, "wb") as f:
            f.write(gzip.decompress(response.read()))

        # Add silence/unknown word specific words.
        with open(lexicon, "a") as f:
            print("<sil> sil", file=f)
            print("<unk> unk", file=f)
    else:
        logging.info(f'lexicon already extracted to {lexicon}')

    units = outdir / "units"
    if not units.is_file():
        unit_set = set()
        with open(lexicon, 'r') as f:
            for line in f:
                for unit in line.strip().split()[1:]:
                    unit_set.add(unit)

        with open(units, 'w') as f:
            for unit in sorted(unit_set):
                print(unit, file=f)

    else:
        logging.info(f'unit set already extracted to {units}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lang', choices=languages, help='language to prepare')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    main(args)
