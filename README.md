# GraphUIL
Our Pytorch implementation of [Graph Neural Networks for User Identity Linkage](https://arxiv.org/pdf/1903.02174).

## 1. Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## 2. Repository Structure
- data/: contains the processed data.
    - graph/: `adj_s.pkl, adj_t.pkl`: adjacency matrices of the source network and the target network.
              `embeds.pkl`: textual input features of two networks.
    - label/: anchor files, train_test_0.x.pkl splits the training anchors at ratios range from 0.1 to 0.9.
    
    The dataset Douban-Weibo is provided by the PHD student Siyuan Chen. If you use the data, please cite the following paper. More details refer to [INFUNE](https://github.com/hilbert9221/INFUNE).
   
     ```
    @inproceedings{chen2020infune,
        title={A Novel Framework with Information Fusion and Neighborhood Enhancement for User Identity Linkage},
        author={Chen, Siyuan and Wang, Jiahai and Du, Xin and Hu, Yanqing},
        booktitle={24th European Conference on Artificial Intelligence (ECAI)},
        pages={1754--1761},
        year={2020}
    }
    ```

- logs/: saving logs 
- models/: contains loss function and metric for evaluation. 
    - base.py
    - loss.py 
    - netEncode.py: GNNs model layers.
- UIL/
    - GraphUIL.py: GraphUIL framework.
- utils/: tool functions for processing data and logging.
- config.py: hyperparameters.
- main.py.

## 3. Runing
```
python main.py
```
