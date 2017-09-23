options="--features chars+embeds+pos"


echo "===== en+es+fr+jp ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.pos.csv data-customer-feedback/en-development.txt.pos.csv --lang en+es+fr+jp $options
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.pos.csv data-customer-feedback/es-development.txt.pos.csv --lang en+es+fr+jp $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.pos.csv data-customer-feedback/fr-development.txt.pos.csv --lang en+es+fr+jp $options
echo "test on JP"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.pos.csv data-customer-feedback/jp-development.txt.pos.csv --lang en+es+fr+jp $options

