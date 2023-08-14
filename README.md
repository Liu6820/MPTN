# MPTN

![ ](MPTN.png)

**This is the data and code for our paper** `MPTN: a message-passing transformer network for drug repurposing from knowledge graph`.

Drug repurposing (DR) based on knowledge graphs (KGs) is challenging, which uses knowledge graph reasoning models to predict new therapeutic pathways for existing drugs. With the rapid development of computing technology and the growing availability of validated biomedical data, various knowledge graph-based methods have been widely used to analyze and process complex and novel data to discover new therapeutic pathways for existing drugs. However, existing methods need to be improved in extracting semantic information from contextual triples of biomedical entities. In this study, we propose a message passing transformer network based on knowledge graph for drug repurposing named MPTN. First, CompGCN is used to precode entity and relation embeddings by jointly aggregating entity and relation embeddings. Then, to fully capture the semantic information of entity context triples, the message passing transformer module is designed. The module integrates the transformer into the message-passing mechanism and incorporates the attention weight information of computing entity context triples into the
entity embedding to update the entity embedding. MPTN utilizes the InteractE module as the decoder to obtain heterogeneous feature interactions in entity and relation representations. Experiments on two datasets show that this model is superior to the existing knowledge graph embedding (KGE) learning methods.

## Prerequisites

* `Python(version >= 3.6)`
* `pytorch(version>=1.4.0)`
*  `ordered_set(version>=3.1)`
* `pytorch>=1.7.1 & <=1.9`
* `numpy(version>=1.16.2)`
* `torch_scatter(version>=2.0.4)`
* `scikit_learn(version>=0.21.1)`
* `torch-geometric`

## Datastes

We provide the dataset in the [data](data/) folder.

- [GP-KG](data/GP-KG/)
- [OpenBioLink](data/openbiolink/)

## Model

The basic structure of our model an be found in [model](model/) folder.
The model can be divided into 4 parts, data loading, CompGCN module, Message passing transformer module and Drug repurposing prediction. They can be used in file [`data_loader.py`](model/data_loader.py), [`CompGCN.py`](model/CompGCN.py), [`MPT.py`](model/MPT.py) and [`predict.py`](model/predict.py).

## Training

Training-related utilities can be found in [`main.py`](main.py). They accept `Iterator`'s that yield batched data,
identical to the output of a `torch.utils.data.DataLoader`. The trained model is saved in the directory “model_saved". 

## Test

Test-related utilities can be found in [`test.py`](test.py). Create a test file named as "drug_pre.txt" and moved the file to the folder "test_data". Predicting results will be saved in the file "results.txt".

