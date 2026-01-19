# Data directory

This repository does **not** ship with any third-party dataset.

## Where to put your dataset
- Put the raw dataset under: `data/raw/<dataset-name>/...`
- Configure `dataset.raw_glob` in your YAML config to match your files.

Example:
```
data/raw/cardiff/...
configs/cardiff.yaml
```

## Licensing & attribution
If your dataset is licensed (e.g., Creative Commons CC BY), you must:
- keep attribution,
- cite the dataset and associated publication,
- follow any usage requirements described by the data provider.

Do **not** commit raw datasets to GitHub if the license or your institution forbids it.
