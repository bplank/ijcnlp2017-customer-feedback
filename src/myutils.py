from nltk.tokenize import TweetTokenizer
import os
import tinysegmenter

def load_data(filename):
    """ loads the data """
    tokenizer = TweetTokenizer() #NLTK tokenizer
    lang=os.path.basename(filename).split("-")[0]
    if lang =="jp":
        import tinysegmenter
        tokenizer = tinysegmenter.TinySegmenter()
    data = [l.strip() for l in open(filename).readlines() if l.strip()]
    sentences = [" ".join(map(str.lower, tokenizer.tokenize(l.split("\t")[1]))) for l in data] # tokenize, map to lowercase
    sentencesPrefixed = [" ".join([lang+":"+w for w in list(map(str.lower, tokenizer.tokenize(l.split("\t")[1])))]) for l in data] # add language prefix
    
    if len(data[0].split("\t")) == 3:
        all_labels = [l.split("\t")[2] for l in data]
    else:
        # use dummy label for test data
        all_labels = ["comment" for l in data] #map to most frequent
    all_sentence_ids =  [l.split("\t")[0] for l in data]
    orgtext = [l.split("\t")[1] for l in data]

    labels = []
    for labs in all_labels:
        labs = labs.strip()
        if "," in labs:
            # for now just take first
            labs = labs.split(",")[0]
        if labs in ["undefined","noneless", "nonsense","undetermined"]:
            labs = "meaningless" # remap, ignore undefined
        labels.append(labs)

    ## make sure we have a label for every data instance
    assert(len(sentences)==len(labels))
    return sentences, labels, all_sentence_ids, orgtext, sentencesPrefixed


if __name__=="__main__":
    import sys
    texts, labels, sids, original_texts, textsPrefix = load_data(sys.argv[1])

    # create pandas dataframe
    import pandas as pd

    df = pd.DataFrame({"labels": labels, "texts": texts, "sentence_ids": sids, "original_texts": original_texts, "textsPrefix":textsPrefix})
    df.to_csv(sys.argv[1]+".csv", index=False)

    # store for fastText
    #for text, label in zip(texts, labels):
    #    print("__label__{}\t{}".format(label, text))




from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
## code from: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class DateStats(BaseEstimator, TransformerMixin):
    """Check whether text contains date"""

    def fit(self, x, y=None):
        return self

    def contains_date(self,string):
        try: 
            parse(string)
            return 1
        except:
            return 0

    def hasNumbers(self, string):
        return any(char.isdigit() for char in string)

    def transform(self, texts):
        return [[self.contains_date(text), self.hasNumbers(text)] for text in texts]

class MeanEmbedding(BaseEstimator, TransformerMixin):
    """get mean embedding vector"""

    def __init__(self, language):
        emb={}
        #file_name = "/home/p252438/corpora/embeds/poly_a/{}.polyglot.txt".format(language) #also skip _UNK 
        #file_name = "/home/p252438/code/fastText_multilingual/vecs/wiki.{}.vec".format(language)
        file_name = "embeds/{}.polyglot.txt".format(language) 
        
        i=0
        for line in open(file_name, errors='ignore', encoding='utf-8'):
            i+=1
            if i==1:
                continue
            try:
                fields = line.strip().split(" ")
                vec = [float(x) for x in fields[1:]]
                word = fields[0]
                emb[word] = vec
            except ValueError:
                print("Error converting: {}".format(line))

        print("loaded pre-trained embeddings {} (word->emb_vec) size: {} ".format(file_name, len(emb.keys())))
        if not "_UNK" in emb:
            emb["_UNK"] = np.ones(len(emb[word])) # add 1's for last word
        self.emb = emb

    def fit(self, x, y=None):
        return self

    def get_mean_emb(self, text):
        """ represent text by mean embedding """
        return np.mean([self.emb.get(w.lower(), self.emb.get("_UNK")) for w in text.split()], axis=0)
#        mean_vec =  np.mean([self.emb.get(w.lower(), self.emb.get("_UNK")) for w in text.split()], axis=0)
#        print(mean_vec)
#        return mean_vec

    def transform(self, texts):
        # returns a dictionary
        return [self.get_mean_emb(text) for text in texts]

