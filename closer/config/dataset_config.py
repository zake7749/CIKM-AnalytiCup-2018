import os

DATASET_ROOT = "../data/dataset/"
ENGLISH_TRAIN_PATH = os.path.join(DATASET_ROOT, "cikm_english_train_20180516.txt")
SPANISH_TRAIN_PATH = os.path.join(DATASET_ROOT, "cikm_spanish_train_20180516.txt")
SPANISH_TEST_PATH = os.path.join(DATASET_ROOT, "cikm_test_a_20180516.txt")
SPANISH_TEST_PATH_SECOND_STAGE = os.path.join(DATASET_ROOT, "cikm_test_b_20180730.txt")

SPANIST_UNLABELED_PATH = os.path.join(DATASET_ROOT, "cikm_unlabel_spanish_train_20180516.txt")

PROCESSED_DATASET_ROOT = "../data/processed_dataset/"
PROCESSED_TRAIN_SET = os.path.join(PROCESSED_DATASET_ROOT, "processed_train.csv")
PROCESSED_TEST_SET = os.path.join(PROCESSED_DATASET_ROOT, "processed_test.csv")
PROCESSED_TRAIN_SET_DROPS_STOPS = os.path.join(PROCESSED_DATASET_ROOT, "processed_train_drops_stopwords.csv")
PROCESSED_TEST_SET_DROPS_STOPS = os.path.join(PROCESSED_DATASET_ROOT, "processed_test_drops_stopwords.csv")
PROCESSED_TRAIN_SET_DROPS_SHARES = os.path.join(PROCESSED_DATASET_ROOT, "processed_train_drops_shares.csv")
PROCESSED_TEST_SET_DROPS_SHARES = os.path.join(PROCESSED_DATASET_ROOT, "processed_test_drops_shares.csv")

AUGMENTED_TRAIN_SET = DATASET_ROOT + "both_augmented.csv"

ENGINEERED_TRAIN_SET = os.path.join(PROCESSED_DATASET_ROOT, "engineered_train.csv")
ENGINEERED_TEST_SET = os.path.join(PROCESSED_DATASET_ROOT, "engineered_test.csv")
ENGINEERED_TRAIN_SET_DROPS_STOPS = os.path.join(PROCESSED_DATASET_ROOT, "engineered_train_drops_stopwords.csv")
ENGINEERED_TEST_SET_DROPS_STOPS = os.path.join(PROCESSED_DATASET_ROOT, "engineered_test_drops_stopwords.csv")
ENGINEERED_TRAIN_SET_DROPS_SHARES = os.path.join(PROCESSED_DATASET_ROOT, "engineered_train_drops_shares.csv")
ENGINEERED_TEST_SET_DROPS_SHARES = os.path.join(PROCESSED_DATASET_ROOT, "engineered_test_drops_shares.csv")

STOPWORD_ROOT = "../data/stopwords/"
SPANISH_STOPWORDS_PATH = os.path.join(STOPWORD_ROOT, "spanishST.txt")

WORDVEC_ROOT = "../data/wordvec/"
ENGLISH_WORDVEC_PATH = os.path.join(WORDVEC_ROOT, "wiki.en.vec")
SPANISH_WORDVEC_PATH = os.path.join(WORDVEC_ROOT, "wiki.es.vec")

ENGLISH_EMBEDDING_SIZE = 300
SPANISH_EMBEDDING_SIZE = 300