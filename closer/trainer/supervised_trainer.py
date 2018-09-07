import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import importlib

from sklearn.metrics import roc_auc_score, log_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint

from closer.config import model_config

class ModelTrainer(object):

    def __init__(self, model_stamp, epoch_num, learning_rate=1e-3,
                 shuffle_inputs=False, verbose_round=40, early_stopping_round=8):
        self.models = []
        self.model_stamp = model_stamp
        self.val_loss = -1
        self.auc = -1
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.eps = 1e-10
        self.verbose_round = verbose_round
        self.early_stopping_round = early_stopping_round
        self.shuffle_inputs = shuffle_inputs
        self.class_weight = [0.93, 1.21]

    def train_translation_folds(self, X, y, fold_count, batch_size, get_model_func, skip_fold=0, patience=10):
        X1, X2, features = X
        fold_id = 9
        models = []
        fold_predictions = []

        train_x1 = X1[:-1400]
        train_x2 = X2[:-1400]
        train_features = features[:-1400]
        train_y = y[:-1400]

        val_x1 = X1[-1400:]
        val_x2 = X2[-1400:]
        val_features = features[-1400:]
        val_y = y[-1400:]

        train_data = {
            "first_sentences": train_x1,
            "second_sentences": train_x2,
            "mata-features": train_features,
        }

        val_data = {
            "first_sentences": val_x1,
            "second_sentences": val_x2,
            "mata-features": val_features,       
        }

        model, bst_val_score, fold_prediction = self._train_model_by_logloss(
            get_model_func(), batch_size, train_data, train_y, val_data, val_y, fold_id, patience)

        models.append(model)
        fold_predictions.append(fold_prediction)

        self.models = models
        self.val_loss = bst_val_score
        return models, bst_val_score, fold_predictions

    def train_folds(self, X, y, fold_count, batch_size, get_model_func, train_chs, augments=None, skip_fold=0, patience=10, scale_sample_weight=False,
                    use_english=False, english_train=None, class_weight=None, self_aware=False, swap_input=False):
        X1, X2, features = X
        train_ch1, train_ch2 = train_chs

        if english_train is not None:
            eX1, eX2 = english_train

        fold_size = len(X1) // fold_count
        models = []
        fold_predictions = []
        score = 0

        for fold_id in range(0, fold_count):
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            if fold_id == fold_count - 1:
                fold_end = len(X1)

            train_x1 = np.concatenate([X1[:fold_start], X1[fold_end:]])
            train_xch1 = np.concatenate([train_ch1[:fold_start], train_ch1[fold_end:]])

            train_x2 = np.concatenate([X2[:fold_start], X2[fold_end:]])
            train_xch2 = np.concatenate([train_ch2[:fold_start], train_ch2[fold_end:]])
            train_features = np.concatenate([features[:fold_start], features[fold_end:]])
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x1 = X1[fold_start:fold_end]
            val_xch1 = train_ch1[fold_start:fold_end]

            val_x2 = X2[fold_start:fold_end]
            val_xch2 = train_ch2[fold_start:fold_end]
            val_features = features[fold_start:fold_end]
            val_y = y[fold_start:fold_end]

            if swap_input:
                r_train_x1, r_train_x2 = train_x1, train_x2
                train_x1 = np.concatenate([r_train_x1, r_train_x2])
                train_x2 = np.concatenate([r_train_x2, r_train_x1])
                train_features = np.concatenate([train_features, train_features])
                train_y = np.concatenate([train_y, train_y])

            if use_english:
                train_ex1 = np.concatenate([eX1[:fold_start], eX1[fold_end:]])
                train_ex2 = np.concatenate([eX2[:fold_start], eX2[fold_end:]])

                train_x1 = np.concatenate([train_x1, train_ex1])
                train_x2 = np.concatenate([train_x2, train_ex2])
                train_features = np.concatenate([train_features, train_features])
                train_y = np.concatenate([train_y, train_y])
            
            if augments is not None:
                a_X1, a_X2, a_label = augments

                def same_order(a, b):
                    if a > b:
                        return (a, b)
                    else:
                        return (b, a)
                dup = set()
                for i in range(len(train_x1)):
                    dup.add(same_order(train_x1[i].tobytes(), train_x1[i].tobytes()))

                aval_ids = []
                for i in range(len(a_X1)):
                    if same_order(a_X1[i].tobytes(), a_X1[i].tobytes()) not in dup: # prevent leakages
                        aval_ids.append(i)

                a_X1 = a_X1[aval_ids]
                a_X2 = a_X2[aval_ids]
                a_label = a_label[aval_ids]

                fake = np.concatenate([train_features, train_features, train_features, train_features])
                fake = fake[:len(a_X1)]
                train_x1 = np.concatenate([train_x1, a_X1])               
                train_x2 = np.concatenate([train_x2, a_X2])   
                train_y = np.concatenate([train_y, a_label])
                train_features = np.concatenate([train_features, fake])

            fold_pos = (np.sum(train_y) / len(train_x1))
            
            if class_weight is not None:
                class_weight = [0.7 / (1-fold_pos), 0.3 / fold_pos]

            weight_val = None
            if scale_sample_weight:
                val_fold_pos = (np.sum(val_y) / len(val_y))
                weight_val = np.ones(len(val_y))
                weight_val *= (0.3 / val_fold_pos)
                weight_val[val_y==0] = (0.7 / (1-val_fold_pos))

            # Double sizes of training and validation sets
            #d_train_x1 = np.concatenate([train_x1, train_x2])
            #d_train_x2 = np.concatenate([train_x2, train_x1])
            #d_train_features = np.concatenate([train_features, train_features])
            #d_train_y = np.concatenate([train_y, train_y])

            train_data = {
                "first_sentences": train_x1,
                "second_sentences": train_x2,
                "mata-features": train_features,
                "first_sentences_char": train_xch1,
                "second_sentences_char": train_xch2,
            }

            val_data = {
                "first_sentences": val_x1,
                "second_sentences": val_x2,
                "mata-features": val_features,       
                "first_sentences_char": val_xch1,
                "second_sentences_char": val_xch2
            }

            model, bst_val_score, fold_prediction = self._train_model_by_logloss(
                get_model_func(), batch_size, train_data, train_y, val_data, val_y, fold_id, patience, class_weight, weight_val)
    
            score += bst_val_score
            models.append(model)
            fold_predictions.append(fold_prediction)

        self.models = models
        self.val_loss = score / fold_count
        return models, self.val_loss, fold_predictions

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id, patience, class_weight, weight_val):
        # return a list with [models, val_loss, oof_predictions]
        raise NotImplementedError

    def _train_model_by_contrastive_loss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id, patience, class_weight, weight_val):
        # return a list with [models, val_loss, oof_predictions]
        raise NotImplementedError

class KerasModelTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(KerasModelTrainer, self).__init__(*args, **kwargs)
        pass

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id, patience, class_weight, weight_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        bst_model_path = self.model_stamp + str(fold_id) + '.h5'
        val_data = (val_x, val_y, weight_val) if weight_val is not None else (val_x, val_y)
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        hist = model.fit(train_x, train_y,
                         validation_data=val_data,
                         epochs=self.epoch_num, batch_size=batch_size, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint],
                         class_weight=class_weight)
        bst_val_score = min(hist.history['val_loss'])
        model.load_weights(bst_model_path)
        predictions = model.predict(val_x)

        return model, bst_val_score, predictions

class PyTorchModelTrainer(ModelTrainer):
    pass