

# Requirements
```bash
git clone [current].git
conda create -y -n revlm python=3.10.0
conda activate revlm
...
pip install -r requirements.txt
```
(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec remove revlm -y
python -m ipykernel install --user --name revlm --display-name "revlm"
```


# Datasets

Download the images from [here](somegoogledrive).

To reproduce the preprocessing of raw aokvqa and fvqa datasets, follow the steps [here](data_raw/README.md). 

Benchmark dataset RationaleVQA (based on AOKVQA, FVQA datasets) can be downloaded from [HuggingFace](https://huggingface.co/datasets/JJoy333/RationaleVQA) 


# Run
```bash
pip install -e .
```
