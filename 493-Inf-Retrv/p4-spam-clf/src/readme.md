# CmpE 493 - Assignment 4

Emre Girgin - 2016400099

### File Structure

- dataset.zip
- eval.py
- feature_extractor.py
- model.py
- naiveBayes.py
- preprocess.py
- readme.md
- report.pdf

### Output Files

- dataset/ (Extracted files)
- dataset.json (Processed dataset)
- model.pth (Pickled Model 1)
- model_mi.pth (Pickled Model 2)

### Preprocess

First, run ```preprocess.py``` and create the processed dataset.

``` bash
python preprocess.py
```

### Naive Bayes Classifier

Run the models and predict.

```bash
python model.py
```

### Test Setup

- Ubuntu 18.04
- Python 3.8.5