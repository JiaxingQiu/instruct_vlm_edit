

# Requirements
```bash
git clone [current].git
conda create -y -n revlm python=3.9.7
conda activate revlm
...
pip install -r requirements.txt
```
(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec remove revlm
python -m ipykernel install --user --name revlm --display-name "revlm"
```


# Datasets


You download imags (and raw AOKVQA, FVQA datasets) following steps [here](data_raw/README.md). 

Benchmark dataset RationaleVQA (based on AOKVQA, FVQA datasets) can be downloaded from [HuggingFace](https://huggingface.co/datasets/JJoy333/RationaleVQA) 
