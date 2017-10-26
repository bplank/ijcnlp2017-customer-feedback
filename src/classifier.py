__author__ = "bplank"

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os

from myutils import ItemSelector, DateStats, MeanEmbedding

seed=103
random.seed(seed)
np.random.seed(seed)

# parse command line options
parser = argparse.ArgumentParser(description="""Simple SVM classifier using various kinds of features (cf. Plank, 2017)""")
parser.add_argument("train", help="train model on a file")
parser.add_argument("test", help="test model on a file")
parser.add_argument("--lang", help="language", default="en")
parser.add_argument("--output", help="output predictions", required=False,action="store_true")
parser.add_argument("--C", help="parameter C for regularization (higher: regularize less)", required=False, default=10, type=float)
parser.add_argument("--num-components", help="svd components", default=40, type=int)
parser.add_argument("--print-confusion-matrix", help="show confusion matrix", action="store_true", default=False)
parser.add_argument("--features", help="feature set", choices=("words","chars","words+chars","embeds", "chars+embeds", "all","all+pos", "chars+embeds+pos"), default="chars+embeds")

args = parser.parse_args()

## read input data
print("load data..")

# using pandas dataframe
df_train = pd.read_csv(args.train)
df_dev = pd.read_csv(args.test)

X_train, y_train = df_train['texts'], df_train['labels']
X_dev, y_dev = df_dev['texts'], df_dev['labels']

labEnc = LabelEncoder()

y_train = labEnc.fit_transform(y_train)
y_dev = labEnc.transform(y_dev)



print("#train instances: {} #dev: {}".format(len(X_train),len(X_dev)))
print("Labels:", labEnc.classes_)



print("vectorize data..")

#algo = LogisticRegression(solver='lbfgs', C=args.C)

algo = LinearSVC(C=args.C)

# tfidf was slightly better than countvectorizer
vectorizerChars = TfidfVectorizer(analyzer='char', ngram_range=(3, 10), binary=True)
vectorizerWords = TfidfVectorizer(ngram_range=(1,2), analyzer='word', binary=True)
vectorizerPos = TfidfVectorizer(ngram_range=(1,3), analyzer='word', binary=True)

if "+" in args.lang:
    embSelector = ItemSelector(key='textsPrefix')
else:
    embSelector = ItemSelector(key='texts')

if args.features == "words":
    features = FeatureUnion([
            ('words',  Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerWords),
            ]))
        ])

elif args.features == "chars":
    features = FeatureUnion([
            ('chars',  Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
        ])

elif args.features == "words+chars":
    features = FeatureUnion([
            # ('words', vectorizerWords),
             #('chars', vectorizerChars),

            ('words',  Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerWords),
            ]))
            ,
            ('chars', Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
        ])
elif args.features == "embeds":
    features = FeatureUnion([
        ('embeds', Pipeline([
        ('selector', embSelector),
                ('mean_emb', MeanEmbedding(args.lang)),
                ('scaler', MinMaxScaler()),
#                ('standardscaler', StandardScaler()),
        ]))
        ])

elif args.features == "chars+embeds": # is the all-in-1 model

    features = FeatureUnion([
            ('chars', Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
            ,
            ('embeds', Pipeline([
                ('selector', embSelector),
                ('mean_emb', MeanEmbedding(args.lang)),
                ('scaler', MinMaxScaler()),
            ]))
   ])


elif args.features == "all":

    features = FeatureUnion([

            ('words',  Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerWords),
            ]))
            ,
            ('chars', Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
            ,
            ('embeds', Pipeline([
                ('selector', embSelector),
                ('mean_emb', MeanEmbedding(args.lang)),
                ('scaler', MinMaxScaler()),

            ]))
   ])

elif args.features == "all+pos":

    features = FeatureUnion([

            ('words',  Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerWords),
            ]))
            ,
            ('chars', Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
            ,
           ('pos',  Pipeline([
                ('selector', ItemSelector(key='pos')),
                ('tfidf', vectorizerPos),
            ]))
            ,
            ('embeds', Pipeline([
                ('selector', embSelector),
                ('mean_emb', MeanEmbedding(args.lang)),
                ('scaler', MinMaxScaler()),

            ]))
   ])

elif args.features == "chars+embeds+pos":

    features = FeatureUnion([


            ('chars', Pipeline([
                ('selector', ItemSelector(key='texts')),
                ('tfidf', vectorizerChars),
            ]))
            ,
           ('pos',  Pipeline([
                ('selector', ItemSelector(key='pos')),
                ('tfidf', vectorizerPos),
            ]))
            ,
            ('embeds', Pipeline([
                ('selector', embSelector),
                ('mean_emb', MeanEmbedding(args.lang)),
                ('scaler', MinMaxScaler()),

            ]))
   ])



classifier = Pipeline([
        ('features', features),
        ('clf', algo)])

print("train model..")

tune=0
debug=0

if tune:
    from sklearn.model_selection import GridSearchCV

    param_grid = {'clf__C': [0.01, 0.02, 0.5, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]}
    grid_search = GridSearchCV(classifier, param_grid, cv=5)

    grid_search.fit(X_train, y_train)
    y_predicted_dev = grid_search.predict(X_dev)
    y_predicted_train = grid_search.predict(X_train)
    print("dev: ", accuracy_score(y_dev, y_predicted_dev))
    print("train: ", accuracy_score(y_train, y_predicted_train))

    print("best:", grid_search.best_params_)
    print("best score:", grid_search.best_score_)


else:
    y_train = df_train['labels']
    y_dev = df_dev['labels']


    classifier.fit(df_train, y_train)

    y_predicted_dev = classifier.predict(df_dev)
    y_predicted_train = classifier.predict(df_train)

    if debug:
        from scipy import stats
        # access weight vectors
        for weights in classifier.named_steps['clf'].coef_:
            print(weights.shape)
            print(stats.describe(weights))

    if args.output:
        # write output
        OUT = open("predictions2/"+os.path.basename(args.test)+"."+os.path.basename(args.train)+"pred.out","w")
        sentence_ids = df_dev['sentence_ids'].values
        org_dev = df_dev['original_texts'].values

        for i, y_pred in enumerate(y_predicted_dev):
            sent_id = sentence_ids[i]
            text = org_dev[i]
        
            OUT.write("{}\t{}\t{}\n".format(sent_id, text, y_pred))
        OUT.close()

    ###

    accuracy_dev = accuracy_score(y_dev, y_predicted_dev)
    accuracy_train = accuracy_score(y_train, y_predicted_train)
    print("Classifier accuracy train: {0:.2f}".format(accuracy_train*100))


    print("===== dev set ====")
    print("Classifier: {0:.2f}".format(accuracy_dev*100))

    mat = confusion_matrix(y_dev, y_predicted_dev)

    if args.print_confusion_matrix:
        sn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=labEnc.classes_, yticklabels=labEnc.classes_)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

    print(classification_report(y_dev, y_predicted_dev, target_names=labEnc.classes_, digits=3))
    f1_dev = f1_score(y_dev, y_predicted_dev, average="weighted")
    print("weighted f1: {0:.1f}".format(f1_dev*100))
    ## end
