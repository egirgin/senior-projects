### Put *model.pickle* in this directory.

to train (**Caution!!!**: training overwrites the existing model)
```bash
python train.py
```
- to switch between lightweight and full implementation, change the ```lightweight``` varible in the **train.py**. The default is **True**
- Ctrl + C stops the training but current model is saved before exiting. This means the existing *model.pickle* will be overwrited.

to evaluate
```commandline
python evaluate.py
```

to train sanity check
```commandline
python sanity.py
```