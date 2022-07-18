#!/usr/bin/env bash

set -eou pipefail


corpusdir=data/corpus
lang="xhosa"
njobs=$(nproc)
ngram_order=3
stage=1
stop_stage=100

. shared/parse_options.sh

manifestsdir=data/manifests/$lang
langdir=data/lang/$lang
graphsdir=data/graphs/$lang

declare -A shortnames=(["xhosa"]="xho" ["sesotho"]="sot" ["setswana"]="tsn" ["zulu"]="zul")


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "================================================================"
    echo " 1. Download and prepare the data"
    echo "================================================================"
    lhotse download soapies -l $lang $corpusdir
    lhotse prepare soapies -l $lang $corpusdir $(dirname $manifestsdir)
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "================================================================"
    echo " 2. Extract features"
    echo "================================================================"

    python local/compute_fbank_soapies.py $lang
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "================================================================"
    echo " 3. Prepare LM and lexicon"
    echo "================================================================"

    mkdir -p data/texts/$lang

    if [ ! -f data/texts/.text.completed ]; then
        pushd data/texts
        wget --no-check-certificate \
            https://material-scripts:ASR_MODELS_STORAGE@data.cstr.ed.ac.uk/material-scripts/jsalt/soapies/text/text.zip
        unzip text.zip
        date > .text.completed
        popd
    fi

    if [ ! -f "data/lm/$lang/lm.${ngram_order}_gram.arpa" ]; then
        python local/prepare_text.py $lang

        echo "estimating ${ngram_order}-gram language model"
        mkdir -p data/lm/$lang
        ngram-count \
            -order $ngram_order \
            -kn-modify-counts-at-end \
            -ukndiscount \
            -gt1min 0 \
            -gt2min 0 \
            -gt3min 0 \
            -text data/texts/$lang/text_train_corpus.txt \
            -lm data/lm/$lang/lm.${ngram_order}_gram.arpa
    else
        echo "${ngram_order}-gram language model already estimated"
    fi

    if [ ! -f data/lang/$lang/lexicon.txt ]; then
        mkdir -p data/lang/$lang
        DR_RULE_DIR=$PWD/nchlt/release/rules perl nchlt/code/pron_predict.pl \
            data/texts/$lang/words.txt \
            ${shortnames[$lang]} \
            data/lang/$lang/rawlexicon.txt

        echo -e "<SIL>\t<SIL>" > data/lang/$lang/lexicon.txt
        echo -e "<UNK>\t<UNK>" >> data/lang/$lang/lexicon.txt
        awk -F'\t' '{print toupper($1)"\t"$2}' \
            data/lang/$lang/rawlexicon.txt \
            >> data/lang/$lang/lexicon.txt

    rm data/lang/$lang/rawlexicon.txt

    fi

    python local/prepare_lang.py --lang-dir $langdir

    if [ ! -f data/lm/$lang/G_${ngram_order}_gram.fst.txt ]; then
        echo "converting arpa lm to fst"
        python3 -m kaldilm \
            --read-symbol-table="data/lang/$lang/words.txt" \
            --disambig-symbol='#0' \
            --max-order=$ngram_order \
            data/lm/$lang/lm.${ngram_order}_gram.arpa > data/lm/$lang/G_${ngram_order}_gram.fst.txt
    else
        echo "lm fst already exists - skipping"
    fi
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

    echo "================================================================"
    echo " 4. Download and prepare the lexicon and the phone set"
    echo "================================================================"

    #python local/download_lexicon.py $lang $langdir
    python local/prepare_lang.py --lang-dir $langdir

fi



if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then

    echo "================================================================"
    echo " 5. Prepare the numerator and denominator graphs"
    echo "================================================================"

    mkdir -p $graphsdir

    cat > $graphsdir/hmm_topo.json <<EOF
{
    "semiring": "LogSemiring{Float32}",
    "initstates": [[1, 0]],
    "arcs": [[1, 2, 0], [2, 2, 0]],
    "finalstates": [[2, 0]],
    "labels": [1, 2]
}
EOF
    echo "HMM topology: $graphsdir/hmm_topo.json"

    cat > $graphsdir/graph_config.toml << EOF
[data]
units = "$langdir/units"
lexicon = "$langdir/lexicon"
train_manifest = "$manifestsdir/soapies-${lang}_supervisions_train.jsonl.gz"
dev_manifest = "$manifestsdir/soapies-${lang}_supervisions_dev.jsonl.gz"

[supervision]
outdir = "$graphsdir"
silword = "<sil>"
unkword = "<unk>"
initial_silprob = 0.8
silprob = 0.2
final_silprob = 0.8
ngram_order = 3
topo = "$graphsdir/hmm_topo.json"
EOF

    echo "graphs (numerator/denominator) config: $graphsdir/graph_config.toml"

    if [ ! -f $graphsdir/.graph.completed ]; then
        CONFIG=$graphsdir/graph_config.toml \
            julia --project=$PWD --procs $njobs local/prepare-lfmmi-graphs.jl
        touch $graphsdir/.graph.completed
    else
        echo "numerator/denominator graphs alreay created"
    fi
fi


