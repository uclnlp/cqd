# Continuous Query Decomposition

---

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complex-query-answering-with-neural-link-1/complex-query-answering-on-fb15k)](https://paperswithcode.com/sota/complex-query-answering-on-fb15k?p=complex-query-answering-with-neural-link-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complex-query-answering-with-neural-link-1/complex-query-answering-on-fb15k-237)](https://paperswithcode.com/sota/complex-query-answering-on-fb15k-237?p=complex-query-answering-with-neural-link-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/complex-query-answering-with-neural-link-1/complex-query-answering-on-nell-995)](https://paperswithcode.com/sota/complex-query-answering-on-nell-995?p=complex-query-answering-with-neural-link-1)


## Updates

- In an [extended abstract for IJCAI](https://www.ijcai.org/proceedings/2022/741),  we have included additional results on the updated query answering datasets introduced with [BetaE](https://arxiv.org/abs/2010.11465). These results are also now on [paperswithcode.com](https://paperswithcode.com/paper/complex-query-answering-with-neural-link-1)!
- We implemented CQD in the [KGReasoning framework](https://github.com/snap-stanford/KGReasoning), a library from [SNAP](http://snap.stanford.edu/) implementing several Complex Query Answering models, which also supports experimenting with the Query2Box and BetaE datasets (in this repo, we only consider the former). Our implementation is available [at this link](https://github.com/pminervini/KGReasoning/).

---

This repository contains the official implementation for our ICLR 2021 (Oral, Outstanding Paper Award) paper, [**Complex Query Answering with Neural Link Predictors**](https://openreview.net/forum?id=Mos9F9kDwkz):

```bibtex
@inproceedings{
    arakelyan2021complex,
    title={Complex Query Answering with Neural Link Predictors},
    author={Erik Arakelyan and Daniel Daza and Pasquale Minervini and Michael Cochez},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Mos9F9kDwkz}
}
```

In this work we present CQD, a method that reuses a pretrained link predictor to answer complex queries, by scoring atom predicates independently and aggregating the scores via t-norms and t-conorms.

Our code is based on an implementation of ComplEx-N3 available [here](https://github.com/facebookresearch/kbc).

Please follow the instructions next to reproduce the results in our experiments.

### 1. Install the requirements

We recommend creating a new environment:

```bash
% conda create --name cqd python=3.8 && conda activate cqd
% pip install -r requirements.txt
```

### 2. Download the data

We use 3 knowledge graphs: FB15k, FB15k-237, and NELL.
From the root of the repository, download and extract the files to obtain the folder `data`, containing the sets of triples and queries for each graph.

```bash
% wget http://data.neuralnoise.com/cqd-data.tgz
% tar xvf cqd-data.tgz
```

### 3. Download the models

Then you need neural link prediction models -- one for each of the datasets.
Our pre-trained neural link prediction models are available here:

```bash
% wget http://data.neuralnoise.com/cqd-models.tgz
% tar xvf cqd-models.tgz
```

### 3. Alternative -- Train your own models

To obtain entity and relation embeddings, we use ComplEx. Use the next commands to train the embeddings for each dataset.

#### FB15k

```bash
% python -m kbc.learn data/FB15k --rank 1000 --reg 0.01 --max_epochs 100  --batch_size 100
```

#### FB15k-237

```bash
% python -m kbc.learn data/FB15k-237 --rank 1000 --reg 0.05 --max_epochs 100  --batch_size 1000
```

#### NELL

```bash
% python -m kbc.learn data/NELL --rank 1000 --reg 0.05 --max_epochs 100  --batch_size 1000
```

Once training is done, the models will be saved in the `models` directory.

### 4. Answering queries with CQD

CQD can answer complex queries via continuous (CQD-CO) or combinatorial optimisation (CQD-Beam).

#### CQD-Beam

Use the `kbc.cqd_beam` script to answer queries, providing the path to the dataset, and the saved link predictor trained in the previous step. For example,

```bash
% python -m kbc.cqd_beam --model_path models/[model_filename].pt
```

Example:

```bash
% PYTHONPATH=. python3 kbc/cqd_beam.py \
  --model_path models/FB15k-model-rank-1000-epoch-100-*.pt \
  --dataset FB15K --mode test --t_norm product --candidates 64 \
  --scores_normalize 0 data/FB15k

models/FB15k-model-rank-1000-epoch-100-1602520745.pt FB15k product 64
ComplEx(
  (embeddings): ModuleList(
    (0): Embedding(14951, 2000, sparse=True)
    (1): Embedding(2690, 2000, sparse=True)
  )
)

[..]
```

This will save a series of JSON fils with results, e.g.

```bash
% cat "topk_d=FB15k_t=product_e=2_2_rank=1000_k=64_sn=0.json"
{
  "MRRm_new": 0.7542805715523118,
  "MRm_new": 50.71081983144581,
  "HITS@1m_new": 0.6896709378392843,
  "HITS@3m_new": 0.7955001359095913,
  "HITS@10m_new": 0.8676865172456019
}
```

#### CQD-CO

Use the `kbc.cqd_co` script to answer queries, providing the path to the dataset, and the saved link predictor trained in the previous step. For example,

```bash
% python -m kbc.cqd_co data/FB15k --model_path models/[model_filename].pt --chain_type 1_2
```

### Final Results

All results from the paper can be produced as follows:

```bash
% cd results/topk
% ../topk-parse.py *.json | grep rank=1000
d=FB15K rank=1000 & 0.779 & 0.584 & 0.796 & 0.837 & 0.377 & 0.658 & 0.839 & 0.355
d=FB237 rank=1000 & 0.279 & 0.219 & 0.352 & 0.457 & 0.129 & 0.249 & 0.284 & 0.128
d=NELL rank=1000 & 0.343 & 0.297 & 0.410 & 0.529 & 0.168 & 0.283 & 0.536 & 0.157
% cd ../cont
% ../cont-parse.py *.json | grep rank=1000
d=FB15k rank=1000 & 0.454 & 0.191 & 0.796 & 0.837 & 0.336 & 0.513 & 0.816 & 0.319
d=FB15k-237 rank=1000 & 0.213 & 0.131 & 0.352 & 0.457 & 0.146 & 0.222 & 0.281 & 0.132
d=NELL rank=1000 & 0.265 & 0.220 & 0.410 & 0.529 & 0.196 & 0.302 & 0.531 & 0.194
```

### Generating explanations

When using CQD-Beam for query answering, we can inspect intermediate decisions.
We provide an example implementation for the case of 2p queries over FB15k-237,
that generates a log file. To generate this log, add the `--explain` flag when running the
`cqd_beam` script. The file will be saved as `explain.log`.

**Note:** for readability, this requires an extra file mapping FB15k-237 entity identifiers
to their original names. Download the file from [this link](https://surfdrive.surf.nl/files/index.php/s/O6yQsBXArSEoYz9/download)
to the `data/FB15k-237` path and untar it.
