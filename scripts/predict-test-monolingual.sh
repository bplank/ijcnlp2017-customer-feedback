options="--output --features chars+embeds"


for lang in en es fr jp
do
    echo "===== $lang ==== "
    python src/classifier.py data-customer-feedback/$lang-training+dev.txt.csv data-customer-feedback/$lang-test.txt.csv --lang $lang $options
done

