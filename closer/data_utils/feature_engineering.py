import pandas as pd
import numpy as np
import re
import gensim
import distance

from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
from gensim import corpora, models
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import ngrams

from closer.config import dataset_config, model_config
from closer.data_utils import bm25
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from simhash import Simhash

minkowski_dis = DistanceMetric.get_metric('minkowski')

class FeatureCreator(object):

    def __init__(self, train_df, test_df, unlabeled_df, data_loader, normalization=True):
        self.train_df = train_df
        self.test_df = test_df
        self.unlabeled_df = unlabeled_df

        self.data_loader = data_loader
        self.stopwords = self.data_loader.load_stopwords()

        self.train_df['splited_spn_1'] = self.train_df['spn_1'].apply(lambda v: v.split())
        self.train_df['splited_spn_2'] = self.train_df['spn_2'].apply(lambda v: v.split())
        self.unlabeled_df['splited_spn_1'] = self.unlabeled_df['spn_1'].apply(lambda v: v.split())

        self.test_df['splited_spn_1'] = self.test_df['spn_1'].apply(lambda v: v.split())
        self.test_df['splited_spn_2'] = self.test_df['spn_2'].apply(lambda v: v.split())

        self.normalization = normalization

        docs = self.train_df['splited_spn_1'].values.tolist() + self.train_df['splited_spn_2'].values.tolist() + \
               self.test_df['splited_spn_1'].values.tolist() + self.test_df['splited_spn_2'].values.tolist() + self.unlabeled_df['splited_spn_1'].tolist()
        docs = np.array(docs)
        docs = np.unique(docs)
        docs = docs.tolist()
        
        docs_raw = self.train_df['spn_1'].values.tolist() + self.train_df['spn_2'].values.tolist() + \
                   self.test_df['spn_1'].values.tolist() + self.test_df['spn_2'].values.tolist() + self.unlabeled_df['spn_1'].tolist()
        docs_raw = np.array(docs_raw)
        docs_raw = np.unique(docs_raw)
        docs_raw = docs_raw.tolist()

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(docs_raw)

        self.bm25_scorer = bm25.bm25Scorer(docs=docs)

        #print("[FE] Loading the word2vec model")
        #self.word2vec_model = KeyedVectors.load_word2vec_format(dataset_config.SPANISH_WORDVEC_PATH)
        #self.word2vec_model.init_sims(replace=True)
        #print("[FE] Loaded the word2vec mdoel")

        self.build_statistic()

    def build_statistic(self):
        self.sentences = self.train_df['splited_spn_1'].tolist() + self.train_df['splited_spn_2'].tolist() + self.test_df['splited_spn_1'].tolist() + self.test_df['splited_spn_2'].tolist() + self.unlabeled_df['splited_spn_1'].tolist()
        self.sentences = np.unique(np.array(self.sentences)).tolist()

        words = []
        for comment in self.sentences:
            for w in comment:
                words.append(w)

        counts = Counter(words)
        self.weights = {word: self._get_weight(count) for word, count in counts.items()}

        self.dictionary = corpora.Dictionary(self.sentences)
        self.dictionary.compactify()
        print ("No of words in the dictionary = %s" % len(self.dictionary.token2id))

    def create_features(self):

        for df in [self.train_df, self.test_df]:
            # create word2vec features
            print("[FE] create the frequency features")
            self._create_frequency_features(df)

            '''Move to preprocessing notebook.
            # create word2vec features
            #print("[FE] create the word2vec features")
            #self._create_word2vec_features(df)
            # create hash features
            #print("[FE] creating the hash features")
            #self._create_hash_features(df)
            #print("[FE] creating the topic features")
            #self._create_topic_features(df)            
            '''

            # create IR features
            print("[FE] creating the IR features")
            self._create_IR_features(df)           

            # create tf/idf weighted distance
            print("[FE] creating the weighted distance features")
            self._create_weighted_distance_features(df)

            # create the length features
            print("[FE] creating the length features")
            self._create_length_features(df)

            # create the meta-information
            print("[FE] creating the weight features")
            self._create_weight_features(df)

            # create the distance features
            print("[FE] creating the distance features")
            self._create_distance_features(df)

            # create fuzzywuzzy features
            print("[FE] creating the fuzzy features")
            self._create_fuzzy_wuzzy_features(df)

            # create topic word features
            print("[FE] creating the topic word features")
            self._create_topic_word_features(df)

            print("[FE] TODO! Create the graph features")
                        
        print("[FE] Feature engineered. With features", self.test_df.columns.values)

        return self.train_df, self.test_df

    def apply_normalization(self, train_df, test_df):
        all_df = pd.concat((train_df, test_df))
        for column in model_config.META_FEATURES:
            if column in all_df.columns:
                scaler = MinMaxScaler()
                all_df[column] = scaler.fit_transform(all_df[column].values.reshape(-1, 1))
            else:
                print("[FE-Norm] The column", column, "is not in the dataframe.")
        train_df, test_df = all_df.iloc[:len(train_df)], all_df.iloc[len(train_df):]
        return train_df, test_df

    def _create_frequency_features(self, df):
        ques = pd.concat([self.train_df[['spn_1', 'spn_2']], \
                          self.test_df[['spn_1', 'spn_2']]], axis=0).reset_index(drop='index')
        q_dict = defaultdict(set)
        for i in range(ques.shape[0]):
                q_dict[ques.spn_1[i]].add(ques.spn_2[i])
                q_dict[ques.spn_2[i]].add(ques.spn_1[i])

        '''Might cause leakage
        def q1_freq(row):
            return(len(q_dict[row['spn_1']]))
            
        def q2_freq(row):
            return(len(q_dict[row['spn_2']]))
        
        def q1_q2_intersect(row):
            return(len(q_dict[row['spn_1']].intersection(q_dict[row['spn_2']])))

        df['q1_q2_intersect'] = df[['spn_1', 'spn_2']].apply(lambda row: q1_q2_intersect(row), axis=1)
        df['q1_freq'] = df[['spn_1', 'spn_2']].apply(lambda row: q1_freq(row), axis=1)
        df['q2_freq'] = df[['spn_1', 'spn_2']].apply(lambda row: q2_freq(row), axis=1)
        '''

    def _create_word2vec_features(self, df):
        df['wmd_distance'] = df[['spn_1', 'spn_2']].apply(lambda row: self.word2vec_model.wmdistance(row['spn_1'], row['spn_2']), axis=1)

    def _create_hash_features(self, df):

        def get_word_ngrams(sequence, n=3):
            return [' '.join(ngram) for ngram in ngrams(sequence, n)]

        def get_character_ngrams(sequence, n=3):
            sequence = ' '.join(sequence)
            return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

        def calculate_simhash_distance(sequence1, sequence2):
            return Simhash(sequence1).distance(Simhash(sequence2))
            
        def calculate_all_simhash(row):
            q1, q2 = row['splited_spn_1'], row['splited_spn_2']
            simhash_distance = calculate_simhash_distance(q1, q2)

            q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
            simhash_distance_2gram = calculate_simhash_distance(q1, q2)

            q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
            simhash_distance_3gram = calculate_simhash_distance(q1, q2)

            q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
            simhash_distance_ch_2gram = calculate_simhash_distance(q1, q2)
           
            q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
            simhash_distance_ch_3gram = calculate_simhash_distance(q1, q2)

            return '{}:{}:{}:{}:{}'.format(simhash_distance, simhash_distance_2gram, simhash_distance_3gram, simhash_distance_ch_2gram, simhash_distance_ch_3gram)

        df['sim_hash'] = df.apply(calculate_all_simhash, axis=1, raw=True)
        df['simhash_distance'] = df['sim_hash'].apply(lambda x: float(x.split(':')[0]))
        df['simhash_distance_2gram'] = df['sim_hash'].apply(lambda x: float(x.split(':')[1]))
        df['simhash_distance_3gram'] = df['sim_hash'].apply(lambda x: float(x.split(':')[2]))
        df['simhash_distance_ch_2gram'] = df['sim_hash'].apply(lambda x: float(x.split(':')[3]))
        df['simhash_distance_ch_3gram'] = df['sim_hash'].apply(lambda x: float(x.split(':')[4]))


    def _create_weighted_distance_features(self, df):
        q1_matrix = self.tfidf_vectorizer.transform(df['spn_1'].values.tolist())
        q2_matrix = self.tfidf_vectorizer.transform(df['spn_2'].values.tolist())
        df['weighted_cosine_sim'] = np.concatenate([cs(q1_matrix[i], q2_matrix[i]).flatten() for i in range(q1_matrix.shape[0])])
        #df['weighted_eucledian_dis'] = np.square((q1_matrix - q2_matrix).toarray()).sum(axis=1)

    def _create_weight_features(self, df):
        df['word_shares'] = df.apply(self._build_word_shares, axis=1, raw=True)

        # weight features
        def first_word_the_same(row):
            return row['splited_spn_1'][0] == row['splited_spn_2'][0]

        df['first_word_the_same'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda row: first_word_the_same(row), axis=1)
        df['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
        df['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
        df['diff_tfidf_word_match'] = (df['word_match'] - df['tfidf_word_match']).abs()
        df['shared_count']  = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
        df['bigram_corr']  = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
        df['trigram_corr']  = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
        df['word_match_no_stopwords'] = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
        df['unique_word_ratio'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda row: len(set(row['splited_spn_1']).union(row['splited_spn_2'])) / (len(row['splited_spn_1']) + len(row['splited_spn_2'])), axis=1)

    def _create_length_features(self, df):

        def word_length_compare(row, cmp):
            l1 = len(row['splited_spn_1'])
            l2 = len(row['splited_spn_2'])
            return cmp(l1, l2)

        def char_length_compare(row, cmp):
            l1 = len(str(row['spn_1']).replace(' ', ''))
            l2 = len(str(row['spn_2']).replace(' ', ''))
            return cmp(l1, l2)

        df['len_word_max'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda v: word_length_compare(v, max), axis=1)
        df['len_word_min'] = df[['splited_spn_2', 'splited_spn_1']].apply(lambda v: word_length_compare(v, min), axis=1)
        df['len_char_max'] = df[['spn_1', 'spn_2']].apply(lambda v: char_length_compare(v, max), axis=1)
        df['len_char_min'] = df[['spn_2', 'spn_1']].apply(lambda v: char_length_compare(v, min), axis=1)

        df['len_word_q1'] = df['splited_spn_1'].apply(len)
        df['len_word_q2'] = df['splited_spn_2'].apply(len)
        df['len_char_q1'] = df['spn_1'].apply(lambda x: len(str(x).replace(' ', '')))
        df['len_char_q2'] = df['spn_2'].apply(lambda x: len(str(x).replace(' ', '')))

        df['word_length_diff'] = (df['len_word_max'] - df['len_word_min']).abs()
        df['char_length_diff'] = (df['len_char_max'] - df['len_char_min']).abs()

        df['len_avg_word_1'] = df['len_word_q1'] / df['len_char_q1']
        df['len_avg_word_2'] = df['len_word_q2'] / df['len_char_q2']
        df['avg_word_diff'] = (df['len_avg_word_1'] - df['len_avg_word_2']).abs()

        def calculate_without_stops_features(row):
            q1_list = row['splited_spn_1']
            q1_set = set(q1_list)
            q1_no_stopwords = q1_set.difference(self.stopwords)

            q2_list = row['splited_spn_2']
            q2_set = set(q2_list)
            q2_no_stopwords = q2_set.difference(self.stopwords)

            return abs(len(q1_no_stopwords) - len(q2_no_stopwords))

        df['len_diff_remove_stopwords'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda v: calculate_without_stops_features(v), axis=1)

    def _create_topic_word_features(self, df):
        
        def add_word_count(df, word):
            df['q1_' + word] = df['splited_spn_1'].apply(lambda x: (word in x) * 1) # * 1 for casting booleans to ints
            df['q2_' + word] = df['splited_spn_2'].apply(lambda x: (word in x) * 1)
            df[word + '_both'] = df['q1_' + word] * df['q2_' + word]

        add_word_count(df, 'cómo') # how
        add_word_count(df, 'qué') # what
        # talk about myself

    def _create_distance_features(self, df):
        q1_csc, q2_csc = self._get_vectors(df, self.dictionary)
        cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = self._get_similarity_values(q1_csc, q2_csc)
        print ("[FE] cosine_sim sample= \n", cosine_sim[0:2])
        print ("[FE] manhattan_dis sample = \n", manhattan_dis[0:2])
        print ("[FE] eucledian_dis sample = \n", eucledian_dis[0:2])
        print ("[FE] jaccard_dis sample = \n", jaccard_dis[0:2])
        print ("[FE] minkowsk_dis sample = \n", minkowsk_dis[0:2])

        eucledian_dis_array = np.array(eucledian_dis).reshape(-1,1)
        manhattan_dis_array = np.array(manhattan_dis).reshape(-1,1)
        minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1,1)

        eucledian_dis = eucledian_dis_array.flatten()
        manhattan_dis = manhattan_dis_array.flatten()
        minkowsk_dis = minkowsk_dis_array.flatten()

        df['cosine_sim'] = cosine_sim
        df['manhattan_dis'] = manhattan_dis
        df['eucledian_dis'] = eucledian_dis
        df['jaccard_dis'] = jaccard_dis
        df['minkowsk_dis'] = minkowsk_dis

    def _create_fuzzy_wuzzy_features(self, df):
        df['fuzzy_ratio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.ratio(row['spn_1'], row['spn_2']), axis=1)
        df['fuzzy_set_ratio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.token_set_ratio(row['spn_1'], row['spn_2']), axis=1)
        df['fuzzy_partial_ratio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.partial_ratio(row['spn_1'], row['spn_2']), axis=1)
        df['fuzzy_token_sort_ratio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.token_sort_ratio(row['spn_1'], row['spn_2']), axis=1)
        df['fuzzy_qratio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.QRatio(row['spn_1'], row['spn_2']), axis=1)
        df['fuzzy_WRatio'] = df[['spn_1', 'spn_2']].apply(lambda row: fuzz.WRatio(row['spn_1'], row['spn_2']), axis=1)
   
        def _get_longest_substr_ratio(a, b):
            strs = list(distance.lcsubstrings(a, b))
            if len(strs) == 0:
                return 0
            else:
                return len(strs[0]) / (min(len(a), len(b)) + 1)

        df['longest_substr_ratio'] = df[['spn_1', 'spn_2']].apply(lambda row: _get_longest_substr_ratio(row['spn_1'], row['spn_2']), axis=1)

    def _create_IR_features(self, df):
        df['bm25_q1_to_q2'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda row: self.bm25_scorer.sim(row['splited_spn_1'], row['splited_spn_2']), axis=1)
        df['bm25_q2_to_q1'] = df[['splited_spn_1', 'splited_spn_2']].apply(lambda row: self.bm25_scorer.sim(row['splited_spn_2'], row['splited_spn_1']), axis=1)

    def _get_vectors(self, df, dictionary):
        question1_vec = [dictionary.doc2bow(text) for text in df.splited_spn_1.tolist()]
        question2_vec = [dictionary.doc2bow(text) for text in df.splited_spn_2.tolist()]
        
        question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
        question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
        
        return question1_csc.transpose(), question2_csc.transpose()

    def _get_similarity_values(self, q1_csc, q2_csc):
        cosine_sim = []
        manhattan_dis = []
        eucledian_dis = []
        jaccard_dis = []
        minkowsk_dis = []
        
        for i,j in zip(q1_csc, q2_csc):
            sim = cs(i, j)
            cosine_sim.append(sim[0][0])
            sim = md(i, j)
            manhattan_dis.append(sim[0][0])
            sim = ed(i, j)
            eucledian_dis.append(sim[0][0])
            i_ = i.toarray()
            j_ = j.toarray()
            try:
                sim = jsc(i_, j_)
                jaccard_dis.append(sim)
            except:
                jaccard_dis.append(0)
                
            sim = minkowski_dis.pairwise(i_, j_)
            minkowsk_dis.append(sim[0][0])
        return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis    

    def _get_weight(self, count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

    def _build_word_shares(self, row):

        q1_list = row['splited_spn_1']
        q1_set = set(q1_list)
        q1_no_stopwords = q1_set.difference(self.stopwords)

        q2_list = row['splited_spn_2']
        q2_set = set(q2_list)
        q2_no_stopwords = q2_set.difference(self.stopwords)
        share_no_stopwords = q1_no_stopwords.intersection(q2_no_stopwords)

        q1words = set(row['splited_spn_1'])
        if len(q1words) == 0:
            return '0:0:0:0:0:0'

        q2words = set(row['splited_spn_2'])
        if len(q2words) == 0:
            return '0:0:0:0:0:0'

        q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
        q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])
        shared_2gram = q1_2gram.intersection(q2_2gram)

        q1_3gram = set(self._generate_ngram(3, q1_list))
        q2_3gram = set(self._generate_ngram(3, q2_list))
        shared_3gram = q1_3gram.intersection(q2_3gram)

        if len(q1_2gram) + len(q2_2gram) == 0:
            R2gram = 0
        else:
            R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))

        if len(q1_3gram) + len(q2_3gram) == 0:
            R3gram = 0
        else:
            R3gram = len(shared_3gram) / (len(q1_3gram) + len(q2_3gram))

        shared_words = q1words.intersection(q2words)
        
        shared_weights = [self.weights.get(w, 0) for w in shared_words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]
        R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
        R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
        return '{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R2gram, R3gram, len(share_no_stopwords))

    def _generate_ngram(self, n, sentence):
        return [tuple(sentence[i:i+n]) for i in range(0, len(sentence) - 1)]