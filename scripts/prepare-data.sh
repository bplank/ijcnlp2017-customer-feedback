for lang in en es fr jp
do
    for f in development training
    do
	python myutils.py data-customer-feedback/$lang-$f.txt
    done
done
