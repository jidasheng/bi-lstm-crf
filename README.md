A PyTorch implementation of the BI-LSTM-CRF model.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/P5P6XKY6B)

# Features:
- Compared with [PyTorch BI-LSTM-CRF tutorial][1], following improvements are performed:
    - Full support for mini-batch computation
    - Full vectorized implementation. Specially, removing all loops in "score sentence" algorithm, which dramatically improve training performance
    - CUDA supported
    - Very simple APIs for [CRF module](#CRF)
        - START/STOP tags are automatically added in CRF
        - A inner Linear Layer is included which transform from feature space to tag space
- Specialized for NLP sequence tagging tasks
- Easy to train your own sequence tagging models
- MIT License

# Installation
- dependencies
    - Python 3
    - [PyTorch][5]
- install
    ```sh
    $ pip install bi-lstm-crf
    ```

# Training
### corpus
- prepare your corpus in the specified [structure and format][2]
- there is also a sample corpus in [`bi_lstm_crf/app/sample_corpus`][3]

### training
```sh
$ python -m bi_lstm_crf corpus_dir --model_dir "model_xxx"
```
- more [options][4]
- [detail of model_dir][7]

### training curve
```python
import pandas as pd
import matplotlib.pyplot as plt

# the training losses are saved in the model_dir
df = pd.read_csv(".../model_dir/loss.csv")
df[["train_loss", "val_loss"]].ffill().plot(grid=True)
plt.show()
```

# Prediction
```python
from bi_lstm_crf.app import WordsTagger

model = WordsTagger(model_dir="xxx")
tags, sequences = model(["市领导到成都..."])  # CHAR-based model
print(tags)  
# [["B", "B", "I", "B", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", "B", "I", "B", "I"]]
print(sequences)
# [['市', '领导', '到', ('成都', 'LOC'), ...]]

# model([["市", "领导", "到", "成都", ...]])  # WORD-based model
```

# <a id="CRF">CRF Module
The CRF module can be easily embeded into other models:
```python
from bi_lstm_crf import CRF

# a BERT-CRF model for sequence tagging
class BertCrf(nn.Module):
    def __init__(self, ...):
        ...
        self.bert = BERT(...)
        self.crf = CRF(in_features, num_tags)

    def loss(self, xs, tags):
        features, = self.bert(xs)
        masks = xs.gt(0)
        loss = self.crf.loss(features, tags, masks)
        return loss

    def forward(self, xs):
        features, = self.bert(xs)
        masks = xs.gt(0)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
```

# Recommendation
- [Video Downloader](https://video-downloader.net): Free online video downloader for thousands of sites 
- [Neural Network](https://github.com/jerryjee120/Neural-Network): A very cool App for Deep Learning & PyTorch


# References
1. [Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional LSTM-CRF Models for Sequence Tagging][6]. arXiv:1508.01991.
2. PyTorch tutorial [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF][1]

[1]:https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
[2]:https://github.com/jidasheng/bi-lstm-crf/wiki/corpus-structure-and-format
[3]:https://github.com/jidasheng/bi-lstm-crf/tree/master/bi_lstm_crf/app/sample_corpus
[4]:https://github.com/jidasheng/bi-lstm-crf/wiki/training-options
[5]:https://pytorch.org/
[6]:https://arxiv.org/abs/1508.01991
[7]:https://github.com/jidasheng/bi-lstm-crf/wiki/details-of-model_dir
