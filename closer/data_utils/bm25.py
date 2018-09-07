import math 

class bm25Scorer(object):

    def __init__(self, docs):

        self.docs = docs
        self.D = len(self.docs)
        self.avgdl = sum([len(doc) for doc in self.docs]) / self.D # average doc length

        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        
        self.initialize()

    def initialize(self, ngram=1):
        for doc in self.docs:
            word_set = set(doc)
            for word in word_set:
                if word not in self.df:
                    self.df[word] = 1
                self.df[word] += 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc1, doc2):
        score = 0
        d = len(doc2)
        loc = self.build_local_df(doc2)
        for word in doc1:
            if word not in loc:
                continue
            score += (self.idf[word] * loc[word] * (self.k1 + 1)
                      / (loc[word] + self.k1 * (1 - self.b + self.b * d
                                                      / self.avgdl)))
        return score

    def build_local_df(self, doc):
        tmp = {}
        for word in doc:
            if not word in tmp:
                tmp[word] = 0
            tmp[word] += 1
        return tmp

    def add_ngram(self,n):
        idx = 0
        for doc in self.docs:
            ngram = self.generate_ngram(n, self.titles[idx])
            seg_list = seg_list + ngram
            idx += 1

    def generate_ngram(self, n, sentence):
        return [sentence[i:i+n] for i in range(0, len(sentence) - 1)]