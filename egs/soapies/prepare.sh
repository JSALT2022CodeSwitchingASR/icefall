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
    echo " 3. Download and prepare the lexicon and the phonet set"
    echo "================================================================"

    python local/prepare_lang.py $lang $langdir

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

    echo "================================================================"
    echo " 4. Prepare LM"
    echo "================================================================"

    python local/prepare_text.py $lang

    mkdir -p data/lm/$lang

    ngram-count \
        -order $ngram_order \
        -kn-modify-counts-at-end \
        -ukndiscount \
        -gt1min 0 \
        -gt2min 0 \
        -gt3min 0 \
        -text data/texts/$lang/text_train+dev.txt \
        -lm data/lm/$lang/lm.${ngram_order}gram.arpa

    if [ ! -f data/lm/$lang/G_${ngram_order}gram.fst.txt ]; then
        python3 -m kaldilm \
            --read-symbol-table="data/lang/$lang/words.txt" \
            --disambig-symbol='#0' \
            --max-order=$ngram_order \
            data/lm/$lang/lm.${ngram_order}gram.arpa > data/lm/$lang/G_${ngram_order}gram.fst.txt
    fi

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


