Run in shell
```
conda create -n edgeai python=3.10
conda activate edgeai

sh env.txt
conda install -c nvidia cuda-nvcc
sudo apt install cmake
pip install bitblas

git clone https://github.com/kingslayer929/edgeai-final
unzip checkpoint-3000.zip

python result.py
```

