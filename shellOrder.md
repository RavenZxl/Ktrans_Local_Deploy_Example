## ubuntu build
```
sudo apt-get update

sudo apt-get upgrade

sudo apt-get install gcc g++ make cmake ninja-build neovim npm
```
## necessary tool install
```
sudo dpkg -i microsoft-edge-stable_134.0.3124.68-1_amd64.deb

sudo dpkg -i code_1.98.2-1741788907_amd64.deb

sudo chmod 775 Anaconda3-2024.10-1-Linux-x86_64.sh

./Anaconda3-2024.10-1-Linux-x86_64.sh

```
## model build
```
pip install modelscope

modelscope download --model unsloth/DeepSeek-R1-GGUF --include '**Q4_K_M**'  --local_dir /root/autodl-tmp/DeepSeek-R1-GGUF
```
## anaconda build
```
conda create --name ktransformers python=3.11
conda activate ktransformers
```
## cuda build

## get ktrans env
```
conda activate ktransformers
```
### torch env
```
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

pip install packaging ninja cpufeature numpy

```
### flash-attn
```
git clone https://github.com/Dao-AILab/flash-attention.git

cd flash-attention

python setup.py install
```

### flashinfer
```
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install -e . -v
```

### get ktrans
```
conda install wheel
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers 
cd website
npm install @vue/cli
npm run build
cd ..
git submodule init 
git submodule update
```
## install ktransformers

```
# export USE_NUMA=1
bash install.sh
```

### test pytorch
```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.get_device_capability())"
export TORCH_CUDA_ARCH_LIST="8.6"
```
## ADDPATH
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_PATH=$CUDA_PATH:/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:$CPATH
export TORCH_CUDA_ARCH_LIST="8.6"
```
## run ktransformers
```
python ./ktransformers/local_chat.py --model_path /media/ravenz/1817-5634/DeepSeek-R1-GGUF --gguf_path /media/ravenz/1817-5634/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL --cpu_infer 70 --max_new_tokens 1024 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml --force_think true --use_cuda_graph 

python ./ktransformers/local_chat.py --model_path /media/ravenz/1817-5634/DeepSeek-R1-GGUF --gguf_path /media/ravenz/1817-5634/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M --cpu_infer 50 --max_new_tokens 1024 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml --force_think true
```
