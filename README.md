# blood_vessel_segmentation

## Data

1. Download:

```shell
kaggle competitions download -c blood-vessel-segmentation
```

2. unzip

unzip and put it in **data** folder

## Env:

```shell
conda create -n {env_name} python=3.10
conda activate {env_name}
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
