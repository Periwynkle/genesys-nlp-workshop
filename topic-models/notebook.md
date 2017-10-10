```python
>>> %matplotlib inline
>>> import matplotlib.pyplot as plt
>>> plt.style.use('seaborn')
>>> import warnings
>>> warnings.filterwarnings('ignore')
>>> import random
>>> random.seed(1)
```

# A Brief Introduction to Topic Modeling with Python

In this session we will employ an unsupervised model of text, one which often goes under the name 'Topic Model', to explore and make visible thematic aspects in the K-Pop comments dataset. Topic Models, or more precisely, mixed-membership models, have gained a lot of popularity as a method for identifying and organizing topical and thematic structures in text documents and text corpora (see e.g. Blei et al 2003, Steyvers & Griffiths 2004). The goal of this session is to introduce the very basics of Topic Models, and, subsequently, to show how Python can be employed to apply Topic Models to the K-Pop dataset. Ignoring any mathematical details, this introduction will primarily focus on the implementation and execution of Topic Models in Python, allowing us to concentrate on the interpretation, evaluation and visualization of the results.

This session is structured as follows. We will first give a brief introduction into the general concepts underlying Topic Modeling. In the subsequent sections, we will work our way through (i) the necessary steps to implement a Topic Model in Python and apply it to the K-Pop dataset, and (ii) means of evaluating and visualizing the results. We conclude with a discussion of a number of Python libraries for Topic Modeling and additional further reading.

## Topic Modeling

## Vector Space Model

```python
>>> import pandas as pd
...
>>> metadata = pd.read_excel("../data/videos.xlsx")
```

```python
>>> import glob
>>> import re
...
>>> ID_re = re.compile(r'.*?_eng-(.*?)_commentsOnly.txt')
...
>>> filepaths = glob.glob('../data/kpop_videos/*/*.txt')
>>> random.shuffle(filepaths)
>>> video_ids = [ID_re.search(path.split('/')[-1]).group(1) for path in filepaths]
>>> group_names = [path.split('/')[-2] for path in filepaths]
```

```python
>>> import sklearn.feature_extraction.text as text
...
>>> vectorizer = text.CountVectorizer(input='filename', min_df=5,
...                                   token_pattern=r'\b[A-Za-z]{3,}\b',
...                                   stop_words='english')
>>> dtm = vectorizer.fit_transform(filepaths)
```

```python
>>> print(f'Shape of document-term matrix: {dtm.shape}. Number of tokens {dtm.sum()}')
Shape of document-term matrix: (202, 36253). Number of tokens 14393178
```

## Topic Modeling K-Pop

---
scrolled: true
...

```python
>>> import lda
...
>>> n_topics = 25
>>> tm = lda.LDA(n_topics=n_topics, n_iter=1500, random_state=1)
>>> document_topic_distributions = tm.fit_transform(dtm)
```

```python
>>> import pandas as pd
...
>>> vocab = vectorizer.get_feature_names()
>>> topic_names = [f'Topic {k}' for k in range(1, n_topics + 1)]
...
>>> topic_word_distributions = pd.DataFrame(
...     tm.components_, columns=vocab, index=topic_names)
...
>>> document_topic_distributions = pd.DataFrame(
...     document_topic_distributions, columns=topic_names, index=video_ids)
```

```python
>>> document_topic_distributions.loc['RuntXwPvvaE'].sort_values(ascending=False).head(10)
Topic 19    0.398471
Topic 24    0.215894
Topic 7     0.075074
Topic 13    0.072437
Topic 20    0.058681
Topic 4     0.055493
Topic 11    0.029449
Topic 16    0.024035
Topic 1     0.023192
Topic 22    0.022457
Name: RuntXwPvvaE, dtype: float64
```

```python
>>> topic_word_distributions.loc['Topic 19'].sort_values(ascending=False).head(10)
come         0.081869
tour         0.021270
bts          0.017781
wings        0.017147
concert      0.015482
money        0.015065
want         0.010863
hope         0.010060
beautiful    0.009773
omg          0.009555
Name: Topic 19, dtype: float64
```

```python
>>> group_topic_distributions = pd.DataFrame(
...     document_topic_distributions.values, index=group_names,
...     columns=topic_names).groupby(level=0).mean()
>>> group_topic_distributions
            Topic 1   Topic 2   Topic 3   Topic 4   Topic 5   Topic 6  \
blackpink  0.004590  0.021518  0.001819  0.113353  0.011846  0.001197   
bts        0.046488  0.033214  0.023785  0.120384  0.015096  0.001873   
exo        0.002825  0.046525  0.000698  0.127615  0.013595  0.025356   
twice      0.005881  0.022312  0.002764  0.114623  0.007739  0.002421   

            Topic 7   Topic 8   Topic 9  Topic 10    ...     Topic 16  \
blackpink  0.024373  0.023372  0.000244  0.043223    ...     0.004886   
bts        0.110502  0.018168  0.000251  0.005521    ...     0.053191   
exo        0.021878  0.042988  0.124437  0.030140    ...     0.000360   
twice      0.012090  0.027114  0.000507  0.013033    ...     0.001022   

           Topic 17  Topic 18  Topic 19  Topic 20  Topic 21  Topic 22  \
blackpink  0.003156  0.045725  0.006849  0.069079  0.002000  0.001722   
bts        0.018634  0.022195  0.017883  0.063160  0.000176  0.018258   
exo        0.048650  0.037220  0.003334  0.070076  0.000077  0.000809   
twice      0.008118  0.061983  0.025374  0.105547  0.354494  0.001434   

           Topic 23  Topic 24  Topic 25  
blackpink  0.334621  0.169242  0.000267  
bts        0.000127  0.211613  0.000165  
exo        0.001657  0.195452  0.102966  
twice      0.005318  0.213573  0.000831  

[4 rows x 25 columns]
```

```python
>>> group_topic_distributions.T.plot.bar()
```

```python
>>> import pyLDAvis
>>> import pyLDAvis.sklearn
...
>>> pyLDAvis.display(pyLDAvis.prepare(**{
...     'vocab': vocab,
...     'doc_lengths': dtm.sum(axis=1).getA1(),
...     'term_frequency': dtm.sum(axis=0).getA1(),
...     'doc_topic_dists': document_topic_distributions.values,
...     'topic_term_dists': topic_word_distributions.values})
>>> )
<IPython.core.display.HTML object>
```

```python
>>> from IPython.display import YouTubeVideo
...
>>> def best_video_for_topic(topic):
...     video_id = document_topic_distributions[topic].argmax()
...     return YouTubeVideo(video_id)
```

```python
>>> best_video_for_topic('Topic 4')
<IPython.lib.display.YouTubeVideo at 0x18469f550>
```

## Exercises

In this

## Further Reading / Package Overview

Gensim, lda, Mallet.