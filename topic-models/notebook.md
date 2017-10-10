```python
>>> %matplotlib inline
>>> import matplotlib.pyplot as plt
>>> plt.style.use('seaborn')
>>> import warnings
>>> warnings.filterwarnings('ignore')
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
>>> import random
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

```python
>>> import sklearn.decomposition
...
>>> n_topics = 25
>>> tm = sklearn.decomposition.LatentDirichletAllocation(
...     n_components=n_topics, learning_method='online', max_iter=50, random_state=1)
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
```

```python
>>> topic_word_distributions.loc['Topic 24'].sort_values(ascending=False).head(10)
```

```python
>>> group_topic_distributions = pd.DataFrame(
...     document_topic_distributions.values, index=group_names,
...     columns=topic_names).groupby(level=0).mean()
>>> group_topic_distributions
```

```python
>>> group_topic_distributions.T.plot.bar()
```

```python
>>> import pyLDAvis
>>> import pyLDAvis.sklearn
...
>>> pyLDAvis.display(pyLDAvis.sklearn.prepare(tm, dtm, vectorizer, mds='tsne'))
```

```python
>>> from IPython.display import YouTubeVideo
...
>>> def best_video_for_topic(topic):
...     video_id = document_topic_distributions[topic].argmax()
...     return YouTubeVideo(video_id)
```

```python
>>> best_video_for_topic('Topic 9')
<IPython.lib.display.YouTubeVideo at 0x11abb5748>
```

## Exercises

In this

## Further Reading / Package Overview

Gensim, lda, Mallet.
