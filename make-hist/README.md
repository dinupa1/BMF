# Building 5D histograms for U-Net training

Install the `miniforge` using the [repository](https://github.com/conda-forge/miniforge).

Use following commands to make conda environment.

```
conda env create --file=environment.yml
conda activate bmf-playground
```

## Apply basic cuts

Basic cuts are applied in [`SimpleTree.py`](SimpleTree.py) file. Modify these cuts as you need. This file will create `simple.root` file which will be using in the net step to create train, validation and test events.

```
python SimpleTree.py
```


## Split to train, validation and test

```
python SplitTree.py
```

## Build the 5D histograms

Use the following commands to build the shared library.

```
root -b

gSystem->CompileMacro("../MakeUNetData.cc", "kfgO", "make-unet-data");

.q
```

The use the following commands to build the histograms.


```
root -b

gSystem->Load("../make-unet-data.so");

MakeUNetData();

.q
```


## Save to tensor

```
python SaveTensor.py
```
