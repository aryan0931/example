# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import clip

KAGGLE_FLAG = False
COLAB_FLAG = False


def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_running_in_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# Check if running in Colab or Kaggle
if is_running_in_colab():
    print("This code is running in Google Colab.")
    from google.colab import drive
    COLAB_FLAG = True
else:
    print("This code is not running in Google Colab.")
    COLAB_FLAG = False

if is_running_in_kaggle():
    print("This code is running in a Kaggle environment.")
    KAGGLE_FLAG = True
else:
    print("This code is not running in a Kaggle environment.")
    KAGGLE_FLAG = False


if COLAB_FLAG:
    # 1. Install the Kaggle library
    os.system("pip install kaggle")

    # 2. Define Kaggle directory
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    kaggle_file_path = os.path.join(kaggle_dir, "kaggle.json")

    # 3. Check if kaggle.json exists in Google Colab secrets or Google Drive
    kaggle_credentials = None
    try:
        from google.colab import userdata
        kaggle_credentials = userdata.get("kaggle_credentials")
    except:
        kaggle_credentials = None

    if kaggle_credentials:
        print("kaggle.json found in Colab Secrets.")
        with open(kaggle_file_path, "w") as file:
            file.write(kaggle_credentials)
    else:
        print("kaggle.json not found in Colab Secrets.")

        # Force mount Google Drive
        drive.mount('/content/drive', force_remount=True)

        # Define path to kaggle.json in Google Drive
        source_path = ""  # Define the correct path here

        # Check if kaggle.json exists in the specified path
        if os.path.exists(source_path):
            shutil.copy(source_path, kaggle_file_path)
            print("kaggle.json found and copied from Google Drive.")
        else:
            print("kaggle.json not found in Google Drive. Please upload it to the correct folder.")

    # 4. Allocate permission to kaggle.json
    os.chmod(kaggle_file_path, 0o600)

# Dataset details and download logic
dataset = "adityajn105/flickr8k"

if COLAB_FLAG:
    kaggle_base = "/content/drive/MyDrive/models/kaggle"
    os.makedirs('/content/drive/MyDrive/models/kaggle/input', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/models/kaggle/working', exist_ok=True)
elif KAGGLE_FLAG:
    kaggle_base = "/kaggle"

kaggle_input = os.path.join(kaggle_base, "input")
kaggle_working = os.path.join(kaggle_base, "working")

# Check if the dataset already exists before downloading
if os.path.exists(os.path.join(kaggle_input, "flickr8k")):
    print("Dataset already exists.")
else:
    # Download and unzip flickr8k dataset
    os.system(f"kaggle datasets download {dataset} -p /content/drive/MyDrive/models/kaggle/input/flickr8k --unzip")

# Unmount Google Drive after completion
if COLAB_FLAG:
    drive.flush_and_unmount()
    print('All changes made in this Colab session are now saved to Google Drive.')

print("Dataset import complete.")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

if COLAB_FLAG:
    drive.mount('/content/drive')

dataset = 'flickr8k'
file_path = os.path.join(kaggle_input, dataset, 'captions.txt')
print(file_path)
df = pd.read_csv(file_path, delimiter=',')
print(df)

df.columns = ['image', 'caption']
df['caption'] = df['caption'].str.lstrip()

os.system("pip install timm")
os.system("pip install ftfy regex tqdm")
os.system("pip install git+https://github.com/openai/CLIP.git")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch version:", torch.__version__, "device", device)

model, preprocess = clip.load("ViT-B/32", device=device)

# Check if CUDA is available and move the model accordingly
if torch.cuda.is_available():
    model.to(device)
model.eval()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# Select a random sample of 12 rows
random_sample = df.sample(n=12, random_state=1)

image_names = random_sample['image'].tolist()
descriptions = random_sample['caption'].tolist()

original_images = []
images = []
texts = []
plt.figure(figsize=(16, 16))
wrapper = textwrap.TextWrapper(width=30)  # Adjust the width to fit your needs
path = kaggle_input
index = 0
for filename in image_names:
    file_path = os.path.join(kaggle_input, dataset, 'Images')
    image = Image.open(os.path.join(file_path, filename)).convert("RGB")

    plt.subplot(3, 4, len(images) + 1)
    plt.imshow(image)

    wrapped_text = wrapper.fill(text=descriptions[index])
    plt.title(f"{filename}\n{wrapped_text}", fontsize=10)

    plt.xticks([])
    plt.yticks([])

    original_images.append(image)

    texts.append(descriptions[index])
    index += 1
    plt.tight_layout()

images = []
for k in original_images:
    # Normalize the image and append to the list
    processed_image = preprocess(k)  # Make sure preprocess is defined and functional
    images.append(processed_image)

# Check if the images list is not empty
if images:
    # Convert the list of preprocessed images to a tensor
    image_input = torch.tensor(np.stack(images)).to(device)

    # Tokenize text input
    text_tokens = clip.tokenize([desc for desc in texts]).to(device)

    # Run the forward pass of the model to get image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
else:
    print("No images were provided to process.")

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

count = len(descriptions)
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)
plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])
plt.title("Cosine similarity between text and image features", size=20)

random_sample = df.sample(n=500, random_state=2)
image_names = random_sample['image'].tolist()
descriptions = random_sample['caption'].tolist()

# Initialize empty lists for images and texts
images = []
texts = []
wrapper = textwrap.TextWrapper(width=30)  # Adjust the width to fit your needs

# Define the file path for images
file_path = os.path.join(kaggle_input, dataset, 'Images')

index = 0
for filename in image_names:
    # Load the image and convert to RGB
    image = Image.open(os.path.join(file_path, filename)).convert("RGB")
    images.append(preprocess(image))  # Ensure 'preprocess' is defined
    texts.append(descriptions[index])
    index += 1

# Convert images to tensor without using CUDA
image_input = torch.tensor(np.stack(images)).float()  # Ensure float type
text_tokens = clip.tokenize([desc for desc in texts])  # Remove .cuda()

# Calculate image and text features
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
count = len(descriptions)

def compute_accuracy(similarity, k, count):
    # Get the indices of the top k matches for each image
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:]

    # Create an array of the correct indices for each text (assuming order matches)
    correct_indices = np.arange(len(texts)).reshape(-1, 1)  # Reshape for broadcasting

    # Check if the correct index is within the top-k for each text
    matches_top_k = np.any(top_k_indices == correct_indices, axis=1)

    # Calculate the top-k accuracy as the proportion of correct matches
    top_k_accuracy = matches_top_k.mean()

    print(f"Evaluating {count} images for Top-{k} Accuracy of matching descriptions to images: {top_k_accuracy:.2f}")

# Call the function for various values of k
compute_accuracy(similarity, 1, count)
compute_accuracy(similarity, 3, count)
compute_accuracy(similarity, 5, count)
compute_accuracy(similarity, 10, count)
compute_accuracy(similarity, 20, count)
compute_accuracy(similarity, 30, count)
