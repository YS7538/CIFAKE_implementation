# CIFAKE: AI-Generated Image Detection

This project compares image classifiers for binary authenticity detection: distinguishing `REAL` images from `FAKE` (AI-generated) images. It evaluates two custom CNNs alongside ImageNet-initialized ResNet-18 and ViT-B/16 models on the local CIFAKE directory present in this workspace.

## Dataset

The local dataset is arranged as `data/CIFAKE/{train,test}/{FAKE,REAL}`. The stored file counts are:

| Split | FAKE | REAL | Total |
|---|---:|---:|---:|
| Train source | 50,000 | 50,000 | 100,000 |
| Test | 10,000 | 10,000 | 20,000 |
| **Total** | **60,000** | **60,000** | **120,000** |

Training code makes a deterministic 90/10 split of the 100,000-image train source (seed 42): 90,000 training and 10,000 validation examples. `ImageFolder` orders the classes as `FAKE` (0) and `REAL` (1). The code also contains a Kaggle-path fallback for `yatinsharma75/image-set`; no documentation or metadata in this repository identifies the original real-image source or which GAN/diffusion generators produced the `FAKE` class.

## Models evaluated

- **Paper CNN** - two 3x3 convolution/ReLU/max-pooling blocks (32 then 32 channels), followed by a 64-unit fully connected layer and a two-class output.
- **Improved CNN** - two convolution/ReLU/max-pooling blocks (32 then 64 channels), followed by a 128-unit fully connected layer and a two-class output.
- **ResNet-18** - Torchvision ResNet-18 initialized with ImageNet default weights; its final fully connected layer is replaced for two classes.
- **ViT-B/16** - Torchvision ViT-B/16 initialized with ImageNet default weights; its classification head is replaced for two classes.

There is no CNN-LSTM model in the repository.

## Reported results

Test metrics below are the results recorded in `experiments.md`; the Improved CNN values also appear in the executed `Notebooks/CNN_improved.ipynb`. Precision, recall, and F1 use scikit-learn’s binary defaults, so the positive class is `REAL` (class 1). Parameter counts are calculated from the model definitions with the two-class heads.

| Model | Test accuracy | Precision | Recall | F1 | Parameters | Logged epochs | Cross-domain accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| Paper CNN | 93.13% | 94.36% | 91.73% | 93.03% | 141,410 | 5 | Not reported |
| Improved CNN | 94.05% | 95.32% | 92.65% | 93.97% | 544,066 | 5 | Not reported |
| ResNet-18 | 96.68% | 95.63% | 97.83% | 96.72% | 11,177,538 | 5 | Not reported |
| ViT-B/16 | 97.15% | 96.87% | 97.45% | 97.16% | 85,800,194 | 1 | Not reported |

No wall-clock training time is recorded. The project does not include a separate cross-domain protocol or result, so the results above should not be represented as GAN-versus-diffusion or cross-generator generalization measurements.

## Explainability

No Grad-CAM, attention-map, saliency, or other explainability implementation is present. The visualization utility plots training/validation loss and validation accuracy only.

## Setup and run

The repository includes a local `data/CIFAKE` directory and a pinned `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`main.py` currently runs the ViT model for one epoch with batch size 64 and 224×224 image resizing. It uses ImageNet default weights, so Torchvision may download them if they are not already cached. Before a local run, change the `checkpoint_path` argument in `main.py`, which is currently hard-coded to `/kaggle/working/vit_b16.pth`, to a writable local path such as `checkpoints/Vit.pth`. The other architectures are model classes and notebook experiments; `main.py` does not provide a command-line selector for them. The custom CNNs are designed for 32×32 input as shown in the improved-CNN notebook.

## Repository structure

```text
.
├── config.py                 # Default batch size, epochs, learning rate, split
├── main.py                   # ViT training/evaluation entry point
├── models/                   # Paper CNN, improved CNN, ResNet-18, and ViT classes
├── utilis/                   # Dataset, training, evaluation, metrics, and plotting helpers
├── Notebooks/                # Improved-CNN notebook and partial CNN/ViT notebooks
├── experiments.md            # Recorded model metrics and confusion matrices
├── checkpoints/              # Local model checkpoints (ignored by Git)
└── data/CIFAKE/              # Local FAKE/REAL train and test image directories (ignored by Git)
```
