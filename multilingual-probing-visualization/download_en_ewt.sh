
# wget https://nlp.stanford.edu/data/ethanchi/multilingual-probing/en.params
# wget https://nlp.stanford.edu/data/ethanchi/multilingual-probing/multiling.params
# mv en.params multiling.params examples/data

wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-test.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-train.conllu

mkdir -p datasets/en/   

mv en_ewt-ud-dev.conllu datasets/en/dev.conllu
mv en_ewt-ud-test.conllu datasets/en/test.conllu
mv en_ewt-ud-train.conllu datasets/en/train.conllu
