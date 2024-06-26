# MonoBERT models for Baidu-ULTR
This repository contains code to train flax-based MonoBERT ranking models from scratch for the large-scale Baidu-ULTR search dataset. The repository is part of a [larger reproducibility study](https://philipphager.github.io/assets/papers/2024-sigir-ultr-meets-reality.pdf) of unbiased learning-to-rank methods on the Baidu-ULTR datasets.

## Setup
1. We recommend installing dependencies using either
     - [Mamba](https://github.com/conda-forge/miniforge): `mamba env create --file environment.yaml`; or
     - [Poetry](https://python-poetry.org/): `poetry install`.
     - Make sure you install [Jax with cuda support for your system](https://jax.readthedocs.io/en/latest/installation.html#installing-jax).
2. Next, download the Baidu ULTR dataset for training. We [upload the first 125 partitions here](https://huggingface.co/datasets/philipphager/baidu-ultr-pretrain/tree/main). Afterwards, update the project config with your dataset path under `config/user_const.yaml`.
3. You can train our BERTs on a SLURM cluster using, e.g.: `sbatch scripts/train.sh <model-name>`, where `<model-name>` is the ranking objective, e.g.: `[naive-pointwise, naive-listwise, pbm, dla, ips-pointwise, ips-listwise]`
5. You can evaluate all pre-trained models by running: `sbatch scripts/eval.sh <model-name>`

## Pre-trained models on HuggingFace Hub
You can download all pre-trained models from hugging face hub by clicking the model names below. We also list the evaluation results on the Baidu-ULTR test set. Ranking performance is measured in DCG, nDCG, and MRR on expert annotations (6,985 queries). Click prediction performance is measured in log-likelihood on one test partition of user clicks (≈297k queries).

| Model                                                                                          | Log-likelihood | DCG@1 | DCG@3 | DCG@5 | DCG@10 | nDCG@10 | MRR@10 |
|------------------------------------------------------------------------------------------------|----------------|-------|-------|-------|--------|---------|--------|
| [Pointwise Naive](https://huggingface.co/philipphager/baidu-ultr_uva-bert_naive-pointwise)     | 0.227          | 1.641 | 3.462 | 4.752 | 7.251  | 0.357   | 0.609  |
| [Pointwise Two-Tower](https://huggingface.co/philipphager/baidu-ultr_uva-bert_twotower)        | 0.218          | 1.629 | 3.471 | 4.822 | 7.456  | 0.367   | 0.607  |
| [Pointwise IPS](https://huggingface.co/philipphager/baidu-ultr_uva-bert_ips-pointwise)         | 0.222          | 1.295 | 2.811 | 3.977 | 6.296  | 0.307   | 0.534  |
| [Listwise Naive](https://huggingface.co/philipphager/baidu-ultr_uva-bert_naive-listwise)       | -              | 1.947 | 4.108 | 5.614 | 8.478  | 0.405   | 0.639  |
| [Listwise IPS](https://huggingface.co/philipphager/baidu-ultr_uva-bert_ips-listwise)           | -              | 1.671 | 3.530 | 4.873 | 7.450  | 0.361   | 0.603  |
| [Listwise DLA](https://huggingface.co/philipphager/baidu-ultr_uva-bert_dla)                    | -              | 1.796 | 3.730 | 5.125 | 7.802  | 0.377   | 0.615  |


## Using pretrained models
```Python
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.data import collate_click_fn
from src.model import CrossEncoder

# As an example, we use a smaller click dataset based on Baidu ULTR:
dataset = load_dataset(
    "philipphager/baidu-ultr_uva-mlm-ctr",
    name="clicks",
    split="test",
    trust_remote_code=True,
)

click_loader = DataLoader(
    test_clicks,
    batch_size=64,
    collate_fn=collate_click_fn,
)

# Download the naive-pointwise model from HuggingFace hub.
# Note that you have to change the model class for instantiating different models:
model = CrossEncoder.from_pretrained(
    "philipphager/baidu-ultr_uva-bert_naive-pointwise",
)

# Use model for click / relevance prediction
batch = next(iter(click_loader))
model(batch)

# Use model only for relevance prediction, e.g., for evaluation:
model.predict_relevance(batch)
```

## Architecture
The basis for all ranking models in this repository is a [MonoBERT cross-encoder architecture](https://arxiv.org/pdf/1910.14424.pdf). In a cross-encoder, the user query and each candidate document are concatenated as the BERT input and the CLS token is used to predict query-document relevance. We train BERT models from scratch using a masked language modeling objective by randomly masking the model input and training the model to predict missing tokens. We tune the CLS token to predict query-document relevance using ranking objectives on user clicks. We display a rough sketch of the model architecture below:

<p align="center">
  <img src='https://github.com/philipphager/baidu-bert-model/assets/9155371/065e704d-51ba-4c2f-ac4d-aa589f44565a' width='600'>
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

## License
This project uses the [MIT License](https://github.com/philipphager/baidu-bert-model/blob/main/LICENSE).
