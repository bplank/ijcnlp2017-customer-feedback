#!/bin/bash
~/code/fastText/fasttext supervised -verbose 2 -input data-customer-feedback/fasttext/ft-en-train.txt -output data-customer-feedback/fasttext/en.model -minn 1 -maxn 9 -dim 100 -lr 0.1 -wordNgrams 2 
~/code/fastText/fasttext predict data-customer-feedback/fasttext/en.model.bin data-customer-feedback/fasttext/ft-en-dev.txt > predictions/ft-en-dev.out
python eval-fasttext.py data-customer-feedback/fasttext/ft-en-dev.txt predictions/ft-en-dev.out
