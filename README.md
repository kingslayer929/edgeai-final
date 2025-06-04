```
conda create -n edgeai python=3.10
conda activate edgeai

git clone https://github.com/kingslayer929/edgeai-final

pip install -r requirements.txt
conda install -c nvidia cuda-nvcc
sudo apt install cmake
cd ~/miniconda/envs/edgeai/lib
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  .
ln -s -f libstdc++.so.6.0.30 libstdc++.so.6
pip install -U bitblas



cd ~/edgeai-final
unzip checkpoint-3000.zip
python result.py
```

