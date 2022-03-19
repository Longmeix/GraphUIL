# GraphUIL
Our Pytorch implementation of Graph Neural Networks for User Identity Linkage.

## 1. Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## 2. Repository Structure
- data/: contains the processed dataset Douban-Weibo, provided by Siyuan Chen
    - graph/: `adj_s.pkl, adj_t.pkl`: adjacency matrices of the source network and the target network.
              `embeds.pkl`: textual input features of two networks.
    - label/: anchor files, train_test_0.x.pkl split the training anchors at ratios range from 0.1 to 0.9.
    
    More details refer to [INFUNE](https://github.com/hilbert9221/INFUNE).
- logs/: saving logs 
- models/: contains loss function and metric for evaluation. 
    - base.py
    - loss.py 
    - netEncode.py: GNNs model layers.
- UIL/
    - GraphUIL.py: GraphUIL framework.
- utils/: tool functions for processing data and logging.
- config.py: hyperparameters.
- main.py

## 3. Runing
```
python main.py
```
