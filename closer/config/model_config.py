USE_CUDA = True
MODEL_CHECKPOINT_FOLDER = "checkpoints/"
TEMPORARY_CHECKPOINTS_PATH = "temporary_checkpoints/"

# TODO Refactor with ArgumentParser

# -- Trainer --
MAX_SENTENCE_LENGTH = 56
MAX_GRADIENT = 10

# -- Word embedding --
WORD_EMBEDDING_SIZE = 300

# -- Meta features --
META_FEATURES_DROPOUT = 0.2

# -- Char CNN --
MAX_WORD_LENGTH = 11
CHAR_VOCAB_SIZE = 100
CHAR_EMBEDDING_SIZE = 10
CHAR_EMBEDDING_FEATURE_MAP_NUMS = [10, 80, 100, 80, 80,]
CHAR_EMBEDDING_WINDOW_SIZES = [1, 2, 3, 4, 5]
CHAR_CNN_OUT_SIZE = sum(CHAR_EMBEDDING_FEATURE_MAP_NUMS)

# -- Compare layer --
COMPARE_LAYER_DROPOUT = 0.2

# -- CARNN --
CARNN_RNN_SIZE = 76
CARNN_AGGREATION_DROPOUT = 0.2
CARNN_COMPARE_LAYER_OUTSIZE = 8
CARNN_COMPARE_LAYER_HIDDEN_SIZE = 288
CARNN_CONCAT_DROPOUT = 0.5
CARNN_MLP_HIDDEN_SIZE = 256

EMBEDDING_DROPOUT = 0.5
CONCAT_DROPOUT = 0.7

# -- DACNN --
DACNN_FUSION_GATE = True
DACNN_FUSION_ACTIVATION = 'relu'
DACNN_AUG_LAYERS = 3
DACNN_AUG_SIZE = 48
DACNN_AUG_ACTIVATION = 'relu'
DACNN_CONCAT_DROPOUT = 0.5

DACNN_NORM = True
DACNN_WITH_BOTTLENECK = True
DACNN_ALL_FEATURES = False
DACNN_HIGHWAY_LAYERS = 2

META_FEATURES = ['bm25_q1_to_q2', 'bm25_q2_to_q1', 'weighted_cosine_sim',
       'len_word_max', 'len_word_min', 'len_char_max', 'len_char_min',
       'word_length_diff', 'char_length_diff', 'len_diff_remove_stopwords',
       'first_word_the_same', 'word_match', 'tfidf_word_match', 'shared_count', 'bigram_corr', 'trigram_corr',
       'word_match_no_stopwords', 'unique_word_ratio', 'cosine_sim',
       'manhattan_dis', 'eucledian_dis', 'jaccard_dis', 'minkowsk_dis',
       'fuzzy_ratio', 'fuzzy_set_ratio', 'fuzzy_partial_ratio',
       'fuzzy_token_sort_ratio', 'fuzzy_qratio', 'fuzzy_WRatio',
       'longest_substr_ratio', 'cómo_both', 'simhash_distance', 'simhash_distance_2gram',
       'simhash_distance_3gram', 'simhash_distance_ch_2gram',
       'simhash_distance_ch_3gram', 'raw_wmd', 'word2vec_jaccard_distance',
       'freq_based_word2vec_cosine_distance', 'freq_based_word2vec_jaccard_distance',
       'lda_balanced_euclidean_distance', 'lsi_cosine_distance',
       'lsi_jaccard_distance', 'jellyfish_jaro_winkler_distance', 'smith_waterman_distance'
]