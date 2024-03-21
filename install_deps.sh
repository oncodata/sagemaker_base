sudo apt update
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install -q slideio==2.1.0
pip install -q opencv-contrib-python==4.5.5.62
pip install -q s3fs
pip install -q torchsummary
pip install -q spams==2.6.5.4
pip install sagemaker --upgrade