# My README for Protenix

## Install
```shell
mamba create -n protenix python=3.12 pip
mamba activate protenix

pip install -r requirements.txt
pip install protenix
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install gemmi==0.6.5 pdbeccdutils==0.8.5

git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass

## Prepare
python scripts/gen_ccd_cache.py -c release_data/ccd_cache/ -n 10
cp -r release_data /home/xux/miniforge3/envs/protenix/lib/python3.12/site-packages/
```

## Run
```shell

# convert
wget https://files.rcsb.org/download/7pzb.cif
mv 7pzb.cif examples/
protenix tojson --input examples/7pzb.cif --out_dir ./output

## Predict
protenix predict --input examples/example.json --out_dir  ./output --seeds 101 --model_name "protenix_base_default_v0.5.0"

## Batch predict
bash inference_demo.sh
```
