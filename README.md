# Icelandic Habitat Classification

Short project note for this repository.

## What this is

The task is to classify Icelandic habitat types (*vistgerðir*) from small satellite image patches. Each sample is a 15×35×35 array (Sentinel‑2 spectral bands + terrain features), and the label is one of 71 habitat classes.

## What I’m doing

I’m keeping the approach simple and reproducible:

- Extract per‑band summary features from each patch (mean, std, min, max, median, and percentiles + normalized band differences).
- Train tree‑based models with fixed random seeds.
- Use **all labeled data** for training (train + validation) and skip a hold‑out validation split, since the competition only allows a single final test submission.

Right now, the best results come from a **Random Forest** trained with **five different seeds** and saved as separate checkpoints.

## How to run

Train five seeded Random Forest models:

```bash
python train_random_forest.py
```
