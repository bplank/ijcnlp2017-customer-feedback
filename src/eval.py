
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

gold = [com.split(",")[0].strip() for com in open(sys.argv[1]).readlines()[1:] if com] #csv file, skip header
pred = [com.split("\t")[2].strip() for com in open(sys.argv[2]).readlines() if com]


print(classification_report(gold, pred, digits=4))
print("weighted F1", f1_score(gold, pred, average="weighted"))
print("micro F1", f1_score(gold, pred, average="micro")*100)
print("accuracy", accuracy_score(gold, pred)*100)

total=0.0
correct=0.0
for p,g in zip(pred,gold):
    if p==g:
        correct+=1
    total+=1
print("accuracy", correct/total*100)
