import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel, Dinov2Model
import argparse
from tqdm.auto import tqdm
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os
import os.path as osp
import cv2 
from torchvision import transforms


def generate_dinov2_embeddings(image_ids, image_paths, model, transform, device, batch_size, regenerate, embeddings_file):
    if not regenerate and embeddings_file.exists():
        return [], np.load(embeddings_file)

    embeddings = []
    damaged_image_ids = []

    with torch.no_grad():
        for i in range(0, len(image_ids), batch_size):
            batch_image_ids = image_ids[i:i+batch_size]
            batch_images = []

            for image_id in batch_image_ids:
                try:
                    image = Image.open(image_paths[image_id]).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)
                    batch_images.append(image)
                except Exception:
                    damaged_image_ids.append(image_id)

            if batch_images:
                batch_images = torch.cat(batch_images, dim=0)
                features = model(batch_images).last_hidden_state[:, 0, :].cpu().numpy()  # CLS token 사용
                embeddings.append(features)

    all_embeddings = np.concatenate(embeddings, axis=0)
    np.save(embeddings_file, all_embeddings)
    return damaged_image_ids, all_embeddings


def get_embeddings(model_dict, device, image_directory, batch_size, regenerate_embeddings, embeddings_file):
    
    allowed_extensions = {".jpeg", ".jpg", ".png", ".webp", '.bmp'}
    images_to_paths, all_image_ids = get_images_to_paths(image_directory, allowed_extensions)
    
    if model_dict['model_name'] == 'clip':
        model = CLIPModel.from_pretrained(model_dict['ckpt']).to(device)
        processor = CLIPProcessor.from_pretrained(model_dict['ckpt'])

        images_to_paths, all_image_ids = get_images_to_paths(image_directory, allowed_extensions)
        damaged_image_ids, all_embeddings = generate_embeddings(all_image_ids, images_to_paths, model, 
                                                                processor, device, batch_size, regenerate_embeddings, embeddings_file)

    elif model_dict['model_name'] == 'dinov2':
        model = Dinov2Model.from_pretrained(model_dict['ckpt']).to(device)
        model.eval()

        # DINOv2 이미지 변환 정의
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        allowed_extensions = {".jpeg", ".jpg", ".png", ".webp", ".bmp"}

        damaged_image_ids, all_embeddings = generate_dinov2_embeddings(
            all_image_ids, images_to_paths, model, transform, device, batch_size, regenerate_embeddings, embeddings_file
        )

    return damaged_image_ids, all_embeddings, all_image_ids, images_to_paths

def process_images(image_directory, model_dict, threshold, batch_size, output_dir, device, offset):
    
    output_dir = osp.join(output_dir, f"{model_dict['model_name']}_{model_dict['ckpt']}", f"offset_{offset}", f"th_{threshold}")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    image_directory = Path(image_directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings_file = image_directory / 'embeddings.npy'
    regenerate_embeddings = check_and_load_embeddings(embeddings_file)

    damaged_image_ids, all_embeddings, all_image_ids, images_to_paths = get_embeddings(model_dict, device, image_directory, batch_size,
                                                       regenerate_embeddings, embeddings_file)

    # if regenerate_embeddings:
    #     np.save(embeddings_file, all_embeddings)

    print("Building Annoy index...")
    annoy_index = build_annoy_index(all_embeddings)

    print("Computing distance matrix...")
    distances = compute_distance_matrix(all_embeddings, annoy_index)

    print("Applying hierarchical clustering...")
    labels = apply_clustering(distances, threshold)
    print(len(np.unique(labels)), len(labels), labels)

    image_id_clusters = build_image_clusters(all_image_ids, labels)
    print(image_id_clusters)

    for idx, image_id_cluster in image_id_clusters.items():
        mosaic = np.zeros((1024, 1024, 3))
        cnt, jdx = 0, 0
        for image in image_id_cluster:
            img = cv2.imread(images_to_paths[image])
            
            img = cv2.resize(img, (256, 256))
            mosaic[int(256*(cnt//4)):int(256*(cnt//4 + 1)), int(256*(cnt%4)):int(256*(cnt%4 + 1))] += img
            cnt += 1
            if cnt > 15:
                cv2.imwrite(osp.join(output_dir, f"{idx}_{jdx}.png"), mosaic)
                jdx += 1
                cnt = 0
                mosaic = np.zeros((1024, 1024, 3))
                
        cv2.imwrite(osp.join(output_dir, f"{idx}_{jdx}.png"), mosaic)
        
def check_and_load_embeddings(embeddings_file):
    # if embeddings_file.exists():
    #     use_existing_embeddings = input("Embeddings file found. Do you want to use existing embeddings? (Y/N) ").strip().lower()
    #     if use_existing_embeddings in ('', 'y', 'yes'):
    #         print("Loading embeddings from file...")
    #         all_embeddings = np.load(embeddings_file)
    #         return False
    # return True
    return True

# Get the paths of all images in the given directory and return the image ids and their paths
def get_images_to_paths(image_directory, allowed_extensions):
    images_to_paths = {
        image_path.stem: image_path
        for image_path in image_directory.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    return images_to_paths, list(images_to_paths.keys())

# Generate CLIP embeddings for all images, handling damaged images if any
def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size, regenerate_embeddings, embeddings_file):
    if not regenerate_embeddings:
        return set(), np.load(embeddings_file)

    damaged_image_ids, all_embeddings = set(), []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating CLIP embeddings")

    for i in range(0, len(all_image_ids), batch_size):
        batch_image_ids, batch_images = process_image_batch(all_image_ids, i, batch_size, images_to_paths, damaged_image_ids)
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        all_embeddings.extend(outputs.cpu().numpy())
        progress_bar.update(len(batch_image_ids))

    progress_bar.close()
    return damaged_image_ids, all_embeddings

# Process a batch of images, returning their ids and loaded images, while identifying damaged images
def process_image_batch(all_image_ids, start_idx, batch_size, images_to_paths, damaged_image_ids):
    batch_image_ids = all_image_ids[start_idx: start_idx + batch_size]
    batch_images = []

    for image_id in batch_image_ids:
        try:
            image = Image.open(images_to_paths[image_id])
            image.load()
            batch_images.append(image)
        except OSError:
            print(f"\nError processing image {images_to_paths[image_id]}, marking as corrupted.")
            damaged_image_ids.add(image_id)

    return batch_image_ids, batch_images

# Build an Annoy index using the generated CLIP embeddings
def build_annoy_index(all_embeddings):
    embeddings = np.array(all_embeddings)
    n_dimensions = embeddings.shape[1]

    annoy_index = AnnoyIndex(n_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(1000)
    return annoy_index

# Compute the distance matrix of the embeddings using the Annoy index
def compute_distance_matrix(all_embeddings, annoy_index):
    n = len(all_embeddings)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            distance = annoy_index.get_distance(i, j)
            distances.append(distance)

    return distances

# Apply hierarchical clustering on the computed distance matrix with the given threshold
def apply_clustering(distances, threshold):
    condensed_distances = np.array(distances)
    Z = linkage(condensed_distances, method='average', optimal_ordering=True)
    return fcluster(Z, t=threshold, criterion='distance')

# Build clusters of image ids based on the clustering labels
def build_image_clusters(all_image_ids, labels):
    image_id_clusters = defaultdict(set)

    for image_id, cluster_label in zip(all_image_ids, labels):
        image_id_clusters[cluster_label].add(image_id)

    return image_id_clusters

def main():

    input_dir = '/HDD/research/clustering/datasets/tenneco_outer/bg_crops'
    output_dir = '/HDD/research/clustering/datasets/tenneco_outer/outputs_bg'
    threshold = 0.7
    batch_size = 1
    device = 'cuda:0'
    # model_dict = {'model_name': 'clip', 'ckpt': 'openai/clip-vit-large-patch14-336', }
    model_dict = {'model_name': 'dinov2', 'ckpt': 'facebook/dinov2-large', }
    offset = 'auto'
    # offset = 100

    process_images(osp.join(input_dir, f'offset_{offset}'), model_dict, threshold, 
                   batch_size, output_dir, device, offset)

if __name__ == "__main__":
    main()