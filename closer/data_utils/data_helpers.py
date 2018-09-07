import pandas as pd
import numpy as np
import re
import gensim

from collections import Counter
from gensim import corpora
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

from closer.config import dataset_config, model_config
from closer.data_utils.tokenizer import StableTokenizer
from closer.data_utils.feature_engineering import FeatureCreator

class DataLoader(object):

    def __init__(self):
        pass

    def load_dataset(self, dataset_path, names):
        '''return a pandas processed csv'''
        return pd.read_csv(dataset_path, names=names, sep='\t')

    def load_clean_words(self, clean_words_path):
        '''return a dict whose key is typo, value is correct word'''
        clean_word_dict = {}
        with open(clean_words_path, 'r', encoding='utf-8') as cl:
            for line in cl:
                line = line.strip('\n')
                typo, correct = line.split(',')
                clean_word_dict[typo] = correct
        return clean_word_dict

    def load_stopwords(self, stopwords_path=dataset_config.SPANISH_STOPWORDS_PATH):
        stopwords = None
        with open(stopwords_path, 'r', encoding='utf-8') as st:
            stopwords = set([w.strip('\n') for w in st])
        return stopwords

    def load_embedding(self, embedding_path):
        '''return a dict whose key is word, value is pretrained word embedding'''
        embeddings_index = {}
        f = open(embedding_path, 'r', encoding='utf-8')
        for line in f:
            values = line.split()
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print("Err on ", values[:2])
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index

class DataTransformer(object):
    
    def __init__(self, max_num_words, max_sequence_length, char_level, focus_on_english=False, features_processed=True):
        self.data_loader = DataLoader()

        self.eng_train_df = self.data_loader.load_dataset(dataset_config.ENGLISH_TRAIN_PATH, names=['eng_1', 'spn_1', 'eng_2', 'spn_2', 'label'])
        self.spn_train_df = self.data_loader.load_dataset(dataset_config.SPANISH_TRAIN_PATH, names=['spn_1', 'eng_1', 'spn_2', 'eng_2', 'label'])
        self.unlabeled_df = self.data_loader.load_dataset(dataset_config.SPANIST_UNLABELED_PATH, names=['spn_1', 'eng_1',])

        # NOTE TO CHANGE TEST SET
        self.old_test_df = self.data_loader.load_dataset(dataset_config.SPANISH_TEST_PATH, names=['spn_1', 'spn_2',])
        self.test_df = self.data_loader.load_dataset(dataset_config.SPANISH_TEST_PATH_SECOND_STAGE, names=['spn_1', 'spn_2',])

        # deprecated! this is a fake file now
        self.augmented_dataset = self.data_loader.load_dataset(dataset_config.SPANISH_TRAIN_PATH, names=['spn_1', 'eng_1', 'spn_2', 'eng_2', 'label'])

        self.train_df = pd.concat([self.eng_train_df, self.spn_train_df]).reset_index(drop=True)
        self.train_df = self.train_df.drop_duplicates()

        self.stopwords = self.data_loader.load_stopwords()
        self.focus_on_english = focus_on_english
        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.max_char_length = model_config.MAX_WORD_LENGTH
        self.char_level = char_level
        self.tokenizer = None
        self.char_tokenizer = None
        self.features_processed = features_processed

        if self.focus_on_english:
            self.language_colunm_1 = 'eng_1'
            self.language_colunm_2 = 'eng_2'
        else:
            self.language_colunm_1 = 'spn_1'
            self.language_colunm_2 = 'spn_2'

        for df in [self.train_df, self.test_df,]:
            df['raw_spn_1'] = df['spn_1'].values
            df['raw_spn_2'] = df['spn_2'].values

        for df in [self.train_df, self.old_test_df, self.test_df, self.augmented_dataset]:
            df['spn_1'] = df['spn_1'].apply(lambda v: self.preprocessing(v))
            df['spn_2'] = df['spn_2'].apply(lambda v: self.preprocessing(v))

        self.unlabeled_df['spn_1'] = self.unlabeled_df['spn_1'].apply(lambda v: self.preprocessing(v))
        self.train_df['eng_1'] = self.train_df['eng_1'].apply(lambda v: self.preprocessing(v))
        self.train_df['eng_2'] = self.train_df['eng_2'].apply(lambda v: self.preprocessing(v))

    def expand_features(self, normalization=True):
        self.feature_creator = FeatureCreator(self.train_df, self.test_df, self.unlabeled_df, data_loader=self.data_loader, normalization=normalization)
        self.train_df, self.test_df = self.feature_creator.create_features()
        self.train_df.to_csv(dataset_config.PROCESSED_TRAIN_SET, index=False, encoding='utf-8')
        self.test_df.to_csv(dataset_config.PROCESSED_TEST_SET, index=False, encoding='utf-8')
        return self.train_df, self.test_df

    def apply_normalization(self, train_df, test_df):
        all_df = pd.concat((train_df, test_df))
        for column in model_config.META_FEATURES:
            if column in all_df.columns:
                scaler = MinMaxScaler()
                all_df[column] = scaler.fit_transform(all_df[column].values.reshape(-1, 1))
            else:
                print("[DH-Norm] The column", column, "is not in the dataframe.")
        train_df, test_df = all_df.iloc[:len(train_df)], all_df.iloc[len(train_df):]
        return train_df, test_df

    def prepare_data(self, drop_stopwords=False, dual=False):
        # TODO Refactor and check
        if not self.features_processed:
            print("[DataHelper Error] Please run the notebook Preprocessing.ipynb before calling prepare_data !!")
            exit()
            #self.train_df, self.test_df = self.expand_features()
        else:
            self.train_df = pd.read_csv(dataset_config.ENGINEERED_TRAIN_SET, encoding='utf-8').fillna(0)
            self.test_df = pd.read_csv(dataset_config.ENGINEERED_TEST_SET, encoding='utf-8').fillna(0)

        # -- get and prepare the sentences --
        sentences_list = []
        for df in [self.train_df, self.augmented_dataset, self.test_df]:
            sentences_list.append(df[self.language_colunm_1].fillna("no comment").values)
            sentences_list.append(df[self.language_colunm_1].fillna("no comment").values)

        sentences_list.append(self.unlabeled_df[self.language_colunm_1].fillna("no comment").values)

        training_labels = self.train_df["label"].values
        train_aug_labels = self.augmented_dataset["label"].values
        
        # -- preprocessing --
        print("Doing preprocessing...")
        processed_sentences_list = []
        for sentences in sentences_list:
            processed_sentences_list.append([self.preprocessing(sentence, drop_stopwords, prefix=None) for sentence in sentences])

        # -- tokenization --
        corpus = [sentence for processed_sentences in processed_sentences_list for sentence in processed_sentences]
        self.build_tokenizer(corpus) # keep the smae order
     
        # -- transform words to ids --
        print("Transforming words to indices...")
        word_idices_list, char_idices_list = [], []
        for sentences in processed_sentences_list[:-1]: # drop the unlabeled sentences
            word_idices_list.append(pad_sequences(self.tokenizer.texts_to_sequences(sentences), maxlen=self.max_sequence_length))
            char_idices_list.append(self.get_padded_char_indices(sentences))

        training_sentence_1, training_sentence_2, train_spanish_aug_1, train_spanish_aug_2, test_sentence_1, test_sentence_2 = word_idices_list
        training_char_sent_1, training_char_sent_2, training_aug_char_sentence_1, training_aug_char_sentence_2, test_char_sent_1, test_char_sent_2 = char_idices_list

        # -- extract meta features --
        train_features = self.train_df[model_config.META_FEATURES].values
        test_features = self.test_df[model_config.META_FEATURES].values

        print('Shape of training data tensor on word level:', training_sentence_1.shape, training_sentence_2.shape)
        print('Shape of training data tensor on char level:', training_char_sent_1.shape, training_char_sent_2.shape)
        print('Shape of testing data tensor on word level:', test_sentence_1.shape, test_sentence_2.shape)
        print('Shape of testing data tensor on char level:', test_char_sent_1.shape, test_char_sent_2.shape)
        print('Shape of label tensor:', training_labels.shape)

        print("Preprocessed.")
        return (training_sentence_1, training_sentence_2, train_features), (test_sentence_1, test_sentence_2, test_features),\
         (training_aug_char_sentence_1, training_aug_char_sentence_2), (train_spanish_aug_1, train_spanish_aug_2, train_aug_labels), \
         (training_char_sent_1, training_char_sent_2), (test_char_sent_1, test_char_sent_2), training_labels

    def preprocessing(self, text, drop_stopwords=False, prefix=None):
        text = text.lower()

        text = re.sub(r";", " ", text)
        text = re.sub(r"’", "'", text)
        text = re.sub(r"‘", "'", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"¿", " ¿ ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"–", " ", text)
        text = re.sub(r"−", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"？", " ? ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"#", " # ", text)
        text = re.sub(r"'", " ", text)

        if prefix is not None:
            text = text.split(' ')
            text = ' '.join([prefix + w for w in text if w != ' '])

        if drop_stopwords:
            text = text.split(' ')
            text = ' '.join([w for w in text if w not in self.stopwords])
        return text

    def build_embedding_matrix(self, embeddings_index):
        nb_words = min(self.max_num_words, len(embeddings_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('null-word.txt', 'w', encoding='utf-8')

        for word, i in word_index.items():

            if i >= self.max_num_words:
                null_words.write(word + '\n')
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + '\n')
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix

    def build_tokenizer(self, comments):
        self.tokenizer = StableTokenizer(num_words=self.max_num_words, char_level=self.char_level)
        self.char_tokenizer = StableTokenizer(num_words=self.max_num_words, char_level=True)
        self.tokenizer.fit_on_texts(comments)
        self.char_tokenizer.fit_on_texts(comments)   

    def get_padded_char_indices(self, sentences):
        processed_squences = self.tokenizer.texts_to_sequences(sentences)
        processed_sentences = self.tokenizer.sequences_to_text(processed_squences)
        all_char_indices = []
        char_to_idx = self.char_tokenizer.word_index

        for sent in processed_sentences:
            char_indices = np.zeros((self.max_sequence_length, self.max_char_length), dtype=np.int32)
            words = sent.split()
            for i in range(self.max_sequence_length):
                if i >= len(words):
                    break
                chars = [c for c in words[i]] 
                for j in range(self.max_char_length):
                    if j >= (len(chars)):
                        break
                    else:
                        index = char_to_idx[chars[j]]
                    char_indices[(self.max_sequence_length - i) - 1, j] = index # revert the order
            all_char_indices.append(char_indices)
        all_char_indices = np.array(all_char_indices)
        return all_char_indices