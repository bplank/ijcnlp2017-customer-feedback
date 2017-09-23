#options="--features all"
options="--features chars+embeds"


echo "===== en+es+fr+jp ==== "
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/en-es-trans.txt.csv --lang en+es+fr+jp $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/en-fr-trans.txt.csv --lang en+es+fr+jp $options
echo "test on JP"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training+dev.txt.csv evaluation/9.customer-feedback-analysis_20170819d/en-jp-trans.txt.csv --lang en+es+fr+jp $options

