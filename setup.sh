#!/bin/bash
apt upgrade & apt update
apt remove python3
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
python3 install_gpu_driver.py
nvidia-smi

# if not formatted, run the below command to format
# mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/{your_disk_name}
mkdir -p /mnt/disk
mount -o auto /dev/{your_disk_name} /mnt/disk
chmod a+w /mnt/disk

# LOTS of stuff will be in ~/.cache
ln -s /mnt/disk/.cache /root/.cache

cd /mnt/disk

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh ./Miniconda3-latest-Linux-x86_64.sh

# don't change. As of 7/25/2023, pytorch has some issues with python 3.11. Just stay with 3.10
conda create -n python310 anaconda python=3.10
conda activate python310
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/artidoro/qlora.git
git clone https://github.com/ggerganov/llama.cpp.git

cd qlora
pip install -r requirements.txt

