# Closer

This is part of 2nd solution manual for CIKM AnalytiCup 2018.

# [Problem Definition](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.187f12b2HAwIBJ&raceId=231661)

The goal of this challenge is to build a cross-lingual short-text matching model. The source language is English and the target language is Spanish. Participants could train their models by applying advanced techniques to classify whether question pairs are the same meaning or not.

# Feature Engineering

There are three kinds of features I used in the final stage.

1. Distance features
I measure the distances between sentence vectors that can be constructed in three ways:

* Bag-of-words model
* Bag-of-words model with TF/IDF
* Weighted average of word embedding based on TF/IDF

2. Topic features
There are many frequent patterns in this dataset. When the customers consult about the same intent, they usually use the same prefix or postfix.
To catch the information of topics, I vectorize the sentences by LDA and LSI
 and compare the difference by cosine distance.

3. Text features
Text features are all about the nature of sentences, for example:
* the length of a sentence: the longer a sentence is, the lower the probability it should be.
* number of stopwords and unique words: it can show the degree of redundancy.
* the variety of words: it usually reflects the complexity of an appeal.

In the end, I create 56 interpretable statistical features. Here is an overview of the feature importance and corelation heatmap.

![feature-importance](https://i.imgur.com/wCD91wt.png)
![feature-heatmap](https://i.imgur.com/vmkO3Hq.png)

# Models

## Decomposable attention

Decomposable attention is a very standard solution to determine if two sentences are the same. In short, decomposable attention aligns the words in sent1 with the similar words in sent2.

![decomposable-attention](https://i.imgur.com/pKxL4Ih.png)
Figure from [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933) 

It helps a lot in this dataset since there are so many surface patterns which repeat again and again. For example:

* What is <X>?
    * What is area number?
    * What is it?
    * What is a wish list?
* How to contact <X>?
    * How to contact the company?
    * How to contact the seller?
* How can I get the <X>?
    * How can I get an order?
    * How can I get my refund?

You can find so many possible choices of <X>, and that's the core issue we are going to address. If we only focus on the statistical features mentioned before, the models would fail in these cases because we encode a word as an identifier and drop its meaning. To be clear, the model would consider "How can I get the order" and "How can I get the refund" the same because they share so many words ("How", "can", "I", "get", "the"). However, a human knows they are not the same due to the difference between "order" and "refund".

You may argue that we can address this issue with TF/IDF, but sometimes the keywords are different but have the same semantics, just like this: 

* I want to talk to a man
* I want to talk to a human
* I want to talk to an employee

These should be considered the same. What the customer truly wants is to talk to a person, it does not matter this person is a woman, a man or an employee.

Decomposable attention compares what it considers related. If two words are indeed related, it's okay. However, if not, it can update the context embedding and make them different. Generally speaking, decomposable attention builds the analogy relationship of each word, which let us catch the significant differences between sentences easily.

Besides decomposable attention, I design the other two models for this task.

### Why not just use decomposable attention?

As I see it, there are two drawbacks of decomposable attention:

1. When applying attention, we discard the order of words, which means we also lose information like phrases or dependencies between words. It could be a disaster as we are prohibited to use any other external data. Once the features are gone, we will never get them back.

2. Cold start words. I found google translator tends to summarize some similar concepts into a single one, which causes the training data to lack variety of words. However, the test data is original Spanish sentences and are not that simple. I guess that's why even though decomposable attention does pretty well on local CV, it always overfits LB hugely.

To deal with these two weaknesses, I dig into another two models. The RNN model is for rebuilding dependencies, and the CNN model is for catching the regional semantic representations. Let's start with the RNN model.

## Compare-Propagare Recurrent Neural Network

In brief, I follow the spirit of Cafe and also propose other ideas that make it work better for this task.

![cafe](https://i.imgur.com/JKU0O4t.png)
Figure from [ A Compare-Propagate Architecture with Alignment Factorization for Natural Language Inference](https://arxiv.org/abs/1801.00102)

1. I replace the FM layer with MLP and mix all interactive features as a single vector. I found it work better when trying to change the FM part with DeepFM. In my opinion, it is because MLP can model the high-level interaction and can be regularized well with dropout.
2. As for the interaction layer, I add the sum operation and compare all kinds of features at the same time. Interestingly, such a simple move boosts the LB score by 0.015.
3. I modify this model to a symmetric architecture. I will talk about the reason in the Regularization section.

## Densely Augmented Convolution Neural Network

When implementing Cafe, I come up with an idea. Why not repeat the loop of augmentation and forwarding, which makes the model fuses local and sequential information much better. I gave it a try. This idea failed for RNN but did succeed for CNN. I think it is because RNN keeps redundant information for each timestep while CNN does not. CNN only focuses on the regional information, and it is also the critical part for this competition.

Let's get back to the first drawback of decomposable attention - dropping the order of words. It is not, for example: 'my' and 'your'. We should not compare these kinds of words independently. What we need is to look at the words near them and compare the whole phrase. Furthermore, this method also mitigates the impact of cold start words. As with the central belief of word2vec, we can comprehend the meaning of a word by its neighbor.

The architecture of DACNN is an extension of decomposable attention. I design a comparison loop and concatenate the results with the original embeddings repeatedly, just like DenseNet. It is an excellent idea for feature reusing and regularization which makes DACNN run 3 times faster than CPRNN with almost the same performance.

We also tried to align the sentence itself by applying the soft alignment twice. It is another style of self-attention which plays a vital role in semantic summarization.

# Model Variety

As we saw in the previous section, these three models are based on different hypotheses. Correlations between each model can also show this fact.

| | CPRNN| DACNN| Decomposable attention|
|--|--|--|--|
| CPRNN | 1 | 0.93 | 0.9 |
| DACNN | 0.93 | 1 | 0.92 |
| Decomposable attention | 0.9|  0.92| 1| 

So I am not going to talk about the differences inter-models. Instead, I am going to show how to create varieties intra-model. The way is straightforward. If we insert different inputs into the same network, the predictions would be different, intuitively. 

I prepare three types of inputs for each model:

1. Word level input. I look up each word from the official pre-trained embeddings. Instead of training the embedding layer directly, I place a highway layer on it to fine-tune the weights.
2. Character level input. I follow the paper "Character-Aware Neural Language Models" to create the character level embedding. Here is the overview of its architecture.

![char-cnn](https://i.imgur.com/r58ISrE.png)
Figure from [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)

4. Word level input with meta-features. The meta-features are well-discussed in Feature Engineering. I concatenate them with the interactive features after a highway layer.

These three inputs do generate different predictions and lead to a significant boost when bagging them.

|Model|Stage 1 LB|
|-------|------------|
|DenseCNN|~ 0.356|
|DenseCharCNN|~0.40|
|DenseMetaCNN|~0.37|
|Bagging 3 Above|~0.343|
|CPRNN|~ 0.353|
|CPCharRNN|~0.40|
|CPMetaRNN|~0.37|
|Bagging 3 Above|~0.34|

You may ask that why don't we take all kinds of inputs at the same time? Indeed, it would be a good try. To be precise, there are 2^3 - 1 combinations we can experiment with. I trained three out of them because I only have one computer. I do believe we can push the performance much further by bagging all types of models.

# Ensemble

Inspired by "Distilling the Knowledge in a Neural Network", I use a two-level stacking for my ensemble. The first level is a simple blending of each type of model. I use the blending predictions as pseudo labels and fine-tune all models with them. 

The second level is DART - [Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866) which takes inputs from fine-tuned models. 

As for pseudo labels, I use the soft labels rather than hard labels for two reasons:

1. The training data is noisy. I can make a simple proof by transitive rule.

* Rule 1: Assume A and B are the same, and B and C are the same. We would expect that A and C are the same.
* Rule 2: Assume A and B are the same, and B and C are not the same. We would expect that A and C are not the same.

Rule 1 and Rule 2 only hold for 75% and 95% cases respectively. I think there is not an absolute rule for labeling and it might differ person by person.

2. The soft label is suitable for regularization and usually more meaningful. There are full discussions about the benefit of soft labels in "Regularizing Neural Networks by Penalizing Confident Output Distributions", and "Learning with Noisy Labels". Especially when taking log-loss as the evaluation metric, we would expect the distribution of outputs to be smoother rather than more extreme. Besides, in a task that does not have a perfect answer, I think to use probability to show the degree of class relationship would make more sense.

# Regularization

## Soft labels

We have discussed this in the Ensemble section.

## Dropout

Instead of setting dropout with small ratio after each dense layer, I tend to use dropout in 2 places:

* Embedding dropout:
Compared to some compora, our training data is way too small. That is, we can not construct a big model or the curse of dimensionality will punish us.
 
So I use embedding dropout that randomly drops some words in the training batch. This move can be considered a kind of data augmentation. With embedding dropout, we can generalize a complicated model like CAFE just using a small amount of data.


* Concat-Feature dropout:
I place the dropout after the interaction layer. It follows the spirit of the random forest because the interaction layer is nothing but a feature pool. The model sometimes focuses on features from sum and multiply, and sometimes selects features from dot and concatenate. This strategy helps the model generalize on different views of data.

## Symmetry

In [Extrapolation in NLP](https://arxiv.org/abs/1805.06648), they observe that a symmetric architecture is a key to generalization. I think it makes sense in this task - the similarity metric is also symmetric. We expect Sim(A, B) should be the same as Sim(B, A). However, we usually do not keep it in mind. We always apply the concatenate operation when creating interactive features and this operation breaks symmetry. 

So how about removing this feature?  Based on the ablation study in Cafe, the concat feature plays a vital role for the model performance.

![ablation-study](https://i.imgur.com/vRD50D8.png)
Table from [ A Compare-Propagate Architecture with Alignment Factorization for Natural Language Inference](https://arxiv.org/abs/1801.00102)

To grasp both symmetry and the concat feature, I do it the concatenate operation twice. The first time is for [Sent1, Sent2] and the second time is for [Sent2, Sent1]. Finally, average these two results.

By the way, I tried other methods to address the issue of symmetry. For example, we can double the training data, one for sentences pair <A, B> and the other for pair <B, A>. Alternatively, we can average the prediction of pair <A, B> and pair <B, A> in the testing phase. However, there are certain costs behind these two solutions, that's why I drop them in the final.

# Performance comparison

|Model| CV logloss |LB logloss|
|-------|-------------|-----------|
|TextCNN |0.31| 0.42|
|TextRNN |0.36 |0.44|
|ESIM w/o ELMO, Syntax tree |0.3 |0.4|
|BiMPM| 0.3 |0.39|
|CAFE |0.24| 0.37|
|Decomposable Attention, word level |0.25 |0.37|
|Decomposable Attention, char level| 0.3| 0.42|
|Decomposable Attention, word level & meta features| **0.22** |0.39|
|CPRNN, word level |0.24| **0.353**|
|CPRNN, char level |0.29| 0.4|
|CPRNN, word level & meta features |0.26 |0.37 |
|DACNN, word level |0.28 |0.356|
|DACNN, char level |0.3 |0.4|
|DACNN, word level & meta features| 0.28| 0.37|
|Blending of DACNN| 0.24 |0.343|
|Blending of CPRNN| 0.21| 0.34|
|Blending of all models| 0.19 |0.33|
|Stacking| 0.17 |No data|

*Note: the log loss of blending methods are measured by corresponding out-of-fold predictions.*

# Reference

1. A Compare-Propagate Architecture with Alignment Factorizationfor Natural Language Inference
2. A Decomposable Attention Model for Natural Language Inference
3. Character-Aware Neural Language Models
4. Convolutional Neural Networks for Sentence Classification
5. DART: Dropouts meet Multiple Additive Regression Trees
6. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
7. Densely Connected Convolutional Networks
8. Distilling the Knowledge in a Neural Network
9. Enhanced LSTM for Natural Language Inference
10. Extrapolation in NLP
11. From Word Embeddings To Document Distances
12. Learning with Noisy Labels
13. Regularizing and Optimizing LSTM Language Models
14. Regularizing Neural Networks by Penalizing Confident Output Distributions
15. R-net: Machine reading comprehension with self matching networks
16. Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm
