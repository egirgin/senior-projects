### Requirements
Any not outdated version of *PyTorch*, *Numpy*, and *MatPlotLib* should be OK.
Make sure you have internet connection to download MNIST dataset. 

## Part1
```bash
cd part1
python train.py
python evaluate.py
```
While training, at each epoch, current *Loss Graph*, *Created Samples*, and *Current VAE Model* will be saved to the current directory.

While evaluating, make sure you have the pickle file of the model at the current directory. Outputs the created samples from fixed noise.


## Part2

Download the Trained Generator from: https://drive.google.com/file/d/1BnXonR0JrHEtUIHaPFXxN2J3NtGUsrxl/view?usp=sharing

to src/part2 directory

```bash
cd part2
python train.py
python evaluate.py
```

While training, at each epoch, current *Loss Graphs*, *Discriminator Predictions Graph*, *Created Samples*, and *Current Generator and Discriminator Model* will be saved to the current directory.

While evaluating, make sure you have the pickle file of the model at the current directory. Outputs the created samples from fixed noise.
