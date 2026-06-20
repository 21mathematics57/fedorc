# FedORC

Implementation of **Communication-Efficient Federated Analytic Class-Incremental
Learning via Orthogonal Residual Compensation**.

FedORC addresses federated class-incremental learning under decentralized and
heterogeneous data. During incremental learning, each client uses only data from
the current phase. Historical samples are neither stored nor replayed. The method
updates lightweight analytic classifiers on top of a frozen backbone and uses an
orthogonal residual compensation stream to improve the fitting capacity of the
classifier.

## Method

FedORC consists of a frozen backbone, a fixed random feature expansion buffer,
a main analytic classifier, and a compensation analytic classifier.

In each phase, clients first solve the main-stream classifier with current-phase
labels. The server aggregates the resulting analytic statistics. Clients then
compute residual targets from the global main classifier. The compensation
stream applies a different activation and projects its features onto the
orthogonal complement of the main-stream feature subspace before solving a
second analytic classifier. The final prediction is obtained by combining the
main-stream and compensation-stream logits.

The paper establishes dual-stream absolute memorization and joint optimality for
the orthogonal dual-stream analytic formulation.

## Repository Layout

```text
FedORC/
  README.md
  requirements.txt
  pyproject.toml
  train.py
  docs/
    FedORC-2.pdf
  src/
    fedorc/
      __init__.py
      aggregation.py
      backbone.py
      buffers.py
      cli.py
      config.py
      data.py
      evaluation.py
      experiment.py
      split.py
      trainer.py
```

The `RandomBuffer` module is included in this repository. No external
`Analytic-continual-learning` checkout is required.

## Installation

```bash
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

## Data

Supported datasets:

- CIFAR-100
- TinyImageNet or ImageNet-style folder datasets

TinyImageNet-style datasets should follow this layout:

```text
TinyImageNet/
  train/
    class_001/
    class_002/
  val/
    class_001/
    class_002/
```

For CIFAR-100, TorchVision downloads the dataset under `--data-root` when
needed.

## Backbone Format

Incremental training uses a frozen backbone. The loader expects
`--backbone-path` to point to a PyTorch checkpoint saved as:

```python
(backbone, _, feature_size)
```

`feature_size` must match `--in-features`.

## Usage

Run from the repository root:

```bash
python train.py \
  --backbone-path /path/to/backbone.pth \
  --data-root /path/to/TinyImageNet \
  --output-dir runs/tinyimagenet \
  --dataset imagenet \
  --num-clients 5 \
  --num-phases 10 \
  --iid-alpha 0.5 \
  --in-features 512 \
  --buffer-size 16384 \
  --main-list 0.5 \
  --com-list 0.14 \
  --comp-list 0.6
```

After editable installation, the console script is also available:

```bash
fedorc-train \
  --backbone-path /path/to/backbone.pth \
  --data-root ./data \
  --dataset cifar100
```

Grid search values are comma-separated:

```bash
python train.py \
  --backbone-path /path/to/backbone.pth \
  --data-root ./data \
  --dataset cifar100 \
  --main-list 0.4,0.5,0.6 \
  --com-list 0.08,0.12,0.16 \
  --comp-list 0.4,0.6,0.8
```

The summary file is saved to:

```text
<output-dir>/summary.csv
```

## Experimental Settings

Settings used in the paper:

| Item | Value |
| --- | --- |
| Clients | 5 |
| Client split | Dirichlet, alpha = 0.5 |
| Backbone | ResNet-18 |
| Base phase | Half of all classes |
| Incremental phases | 5, 10, 25, 50 |

Hyperparameters:

| Dataset | Main lambda | Compensation lambda | Buffer size |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 0.6 | 0.12 | 8192 |
| TinyImageNet | 0.5 | 0.14 | 16384 |

## Reported Results

| Dataset | Best average accuracy | Forgetting range |
| --- | ---: | ---: |
| CIFAR-100 | 0.7348 | 0.0336-0.0346 |
| TinyImageNet | 0.6142 | 0.0579-0.0615 |

FedORC is compared with TARGET, MFCL, LANDER, DCFCL, and AFCL under
`K = 5, 10, 25, 50`. The reported communication volume on CIFAR-100 ranges from
`1.901 GB` at `K = 5` to `4.112 GB` at `K = 50`.

## Notes

- Backbone pre-training is not included.
- The current implementation uses `--comp-list` as a fixed compensation-logit
  fusion coefficient.
- The paper PDF is stored at `docs/FedORC-2.pdf`.

## Citation

```bibtex
@article{wu2026fedorc,
  title={Communication-Efficient Federated Analytic Class-Incremental Learning via Orthogonal Residual Compensation},
  author={Wu, Shanghao and Wang, Yongkang and Zhang, Hai},
  year={2026}
}
```
