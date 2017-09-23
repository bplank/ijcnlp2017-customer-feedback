for file in data-customer-feedback/*development.txt; do python src/myutils.py $file; done
for file in data-customer-feedback/*training.txt; do python src/myutils.py $file; done
for file in data-customer-feedback/*test.txt; do python src/myutils.py $file; done
