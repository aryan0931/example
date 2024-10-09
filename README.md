# Fixed Quick Guide for Singletask Learning Bench

## Step.1 Ianvs preparation

```bash
# Create a new conda environment
conda create -n ianvs-py36 python=3.6
conda activate ianvs-py36

# Clone path-error-fixed Ianvs
git clone https://github.com/FuryMartin/ianvs.git
# switch to quickstart branch
git switch quickstart 
cd ianvs

# Download dependency-error-fixed Sedna
wget https://github.com/FuryMartin/sedna/releases/download/v0.4.1.1/sedna-0.4.1.1-py3-none-any.whl
pip install sedna-0.4.1.1-py3-none-any.whl

pip install -r requirements.txt

```

## Step.2 Example Environment Preparation

```bash
pip install examples/resources/algorithms/FPN_TensorFlow-0.1-py3-none-any.whl

# install cv2(opencv-python)
sudo apt-get update
sudo apt-get install libgl1-mesa-glx  # This is needed by cv2(opencv-python)

pip install opencv-python~=3.4 # must be under 4.0, otherwise it will build from source with a long time
```

## Step.3 Dataset and Model Preparation

```bash
# be sure to be in ianvs root path

mkdir dataset
cd dataset
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip dataset.zip

cd .. # back to ianvs root path

mkdir initial_model
cd initial_model
git clone https://github.com/openai/CLIP.git
```

## Step.4 Run 

```bash
$ python benchmarking.py -f ./examples/pcb-aoi/singletask_learning_bench/fault detection/benchmarkingjob.yaml


