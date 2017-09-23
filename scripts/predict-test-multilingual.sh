options="--output --features chars+embeds"


echo "===== en+es+fr+jp ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv data-customer-feedback/en-test.txt.csv --lang en+es+fr+jp $options
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv data-customer-feedback/es-test.txt.csv --lang en+es+fr+jp $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv data-customer-feedback/fr-test.txt.csv --lang en+es+fr+jp $options
echo "test on JP"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv data-customer-feedback/jp-test.txt.csv --lang en+es+fr+jp $options

