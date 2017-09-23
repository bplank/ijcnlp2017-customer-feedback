#options="--features all"
options="--features chars+embeds"


echo "===== en+es+fr+jp ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/en-test-oracle.txt.csv --lang en $options
echo "test on ES"
python src/classifier.py data-customer-feedback/es-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/es-test-oracle.txt.csv --lang es $options
echo "test on FR"
python src/classifier.py data-customer-feedback/fr-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/fr-test-oracle.txt.csv --lang fr $options
echo "test on JP"
python src/classifier.py data-customer-feedback/jp-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/jp-test-oracle.txt.csv --lang jp $options

