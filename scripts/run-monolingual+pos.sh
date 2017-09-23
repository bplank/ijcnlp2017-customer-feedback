#options="--features $1"
options="--features chars+embeds+pos"

for lang in en es fr jp
do
    echo "===== $lang ==== "
    python src/classifier.py data-customer-feedback/$lang-training.txt.pos.csv data-customer-feedback/$lang-development.txt.pos.csv --lang $lang $options
done
