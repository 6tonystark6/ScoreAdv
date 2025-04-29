import os
import random
import shutil

source_dir = './lfw'
src_output_dir = './lfw_sample/src'
tgt_output_dir = './lfw_sample/target'

os.makedirs(src_output_dir, exist_ok=True)
os.makedirs(tgt_output_dir, exist_ok=True)

all_images = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(os.path.join(root, file))

if len(all_images) < 200:
    raise ValueError("not enough figures, please check the dataset")

selected_images = random.sample(all_images, 200)
src_images = selected_images[:100]
tgt_images = selected_images[100:]

for idx, img in enumerate(src_images, start=1):
    shutil.copy(img, os.path.join(src_output_dir, f"{idx}.jpg"))

for idx, img in enumerate(tgt_images, start=101):
    shutil.copy(img, os.path.join(tgt_output_dir, f"{idx}.jpg"))

print("copied successful")
