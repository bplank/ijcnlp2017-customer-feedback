options="--features $1"
#options="--features chars+embeds"
#options="--features all"
for lang in en es fr jp
do
    echo "===== $lang ==== "
    python src/classifier.py data-customer-feedback/$lang-training.txt.csv data-customer-feedback/$lang-development.txt.csv --lang $lang $options
done
