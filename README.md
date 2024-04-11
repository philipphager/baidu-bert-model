# MonoBERT models for Baidu-ULTR
This repository contains code to train flax-based MonoBERT ranking models from scratch for the large-scale Baidu-ULTR search dataset. The repository is part of a [larger reproducibility study](https://philipphager.github.io/assets/papers/2024-sigir-ultr-meets-reality.pdf) of unbiased learning-to-rank methods on the Baidu-ULTR datasets.

## Setup
1. We recommend installing dependencies using [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html): `mamba env create --file environment.yaml`
2. Next, download the Baidu ULTR dataset for training. We [upload the first 125 partitions here](https://huggingface.co/datasets/philipphager/baidu-ultr-pretrain/tree/main). Afterwards, update project config with your dataset path under `config/user_const.yaml`.
3. Lastly, you can find the training scripts used for training our BERTs on SLURM under `scipts/` and you can run them, e.g., using: `sbatch scripts/cross-encoder.sh`

## Download pretrained models
```

```

## Architecture
The basis for all ranking models in this repository is a [MonoBERT cross-encoder architecture](https://arxiv.org/pdf/1910.14424.pdf). In a cross-encoder, the user query and each candidate document are concatenated as the BERT input and the CLS token is used to predict query-document relevance. We train BERT models from scratch using a masked language modeling objective by randomly masking the model input and training the model to predict missing tokens. We tune the CLS token to predict query-document relevance using ranking objectives on user clicks. We display a rough sketch of the model architecture below:

<p align="center">
  <img src='https://github.com/philipphager/baidu-bert-model/assets/9155371/2c0a6c09-a9c5-4e09-bd1a-d7af9daac079' width='600'>
</p>

We use a pointwise binary cross-entropy loss and a listwise softmax cross-entropy loss as our main ranking losses. We implement several unbiased learning to rank methods for position bias mitigation in click data, including a Two-Tower/PBM objective, inverse propensity scoring (IPS), and the dual learning algorithm (DLA). For more details see our paper or inspect our loss functions at `src/loss.py`.

## Reference
```
@inproceedings{Hager2024BaiduULTR,
  author = {Philipp Hager and Romain Deffayet and Jean-Michel Renders and Onno Zoeter and Maarten de Rijke},
  title = {Unbiased Learning to Rank Meets Reality: Lessons from Baidu’s Large-Scale Search Dataset},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`24)},
  organization = {ACM},
  year = {2024},
}
```
