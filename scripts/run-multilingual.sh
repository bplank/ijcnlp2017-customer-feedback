#options="--features all"
options="--features chars+embeds"


echo "===== en+es ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+es-training.txt.csv data-customer-feedback/en-development.txt.csv --lang en+es $options
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es-training.txt.csv data-customer-feedback/es-development.txt.csv --lang en+es $options

echo "===== en+fr ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+fr-training.txt.csv data-customer-feedback/en-development.txt.csv --lang en+fr $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+fr-training.txt.csv data-customer-feedback/fr-development.txt.csv --lang en+fr $options


echo "===== en+jp ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+jp-training.txt.csv data-customer-feedback/en-development.txt.csv --lang en+jp $options
echo "test on JP"
python src/classifier.py data-customer-feedback/en+jp-training.txt.csv data-customer-feedback/jp-development.txt.csv --lang en+jp $options

echo "===== en+es+fr ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+es+fr-training.txt.csv data-customer-feedback/en-development.txt.csv --lang en+es+fr $options
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es+fr-training.txt.csv data-customer-feedback/es-development.txt.csv --lang en+es+fr $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+es+fr-training.txt.csv data-customer-feedback/fr-development.txt.csv --lang en+es+fr $options

echo "===== en+es+fr+jp ==== "
echo "test on EN"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.csv data-customer-feedback/en-development.txt.csv --lang en+es+fr+jp $options
echo "test on ES"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.csv data-customer-feedback/es-development.txt.csv --lang en+es+fr+jp $options
echo "test on FR"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.csv data-customer-feedback/fr-development.txt.csv --lang en+es+fr+jp $options
echo "test on JP"
python src/classifier.py data-customer-feedback/en+es+fr+jp-training.txt.csv data-customer-feedback/jp-development.txt.csv --lang en+es+fr+jp $options

