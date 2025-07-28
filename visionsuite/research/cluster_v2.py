import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp

# Step 1: Preprocess images and extract features

def preprocess_and_extract_features(image_folder, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    image_paths = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                feature = model(input_tensor)
            
            features.append(feature.squeeze().numpy())
            image_paths.append(image_path)

    return np.array(features), image_paths

# Step 2: Apply PCA for dimensionality reduction

def apply_pca(features, n_components=18):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# Step 3: Perform K-means clustering

def perform_kmeans(features, max_clusters=10):
    best_n_clusters = 2
    best_score = -1

    for n_clusters in range(2, min(max_clusters + 1, len(features))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    return kmeans.fit_predict(features)

def select_representatives(features, cluster_labels, image_paths, n_representatives=3):
    representatives = []
    for cluster in range(max(cluster_labels) + 1):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_features = features[cluster_indices]
        cluster_center = np.mean(cluster_features, axis=0)
        
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
        sorted_indices = np.argsort(distances)
        
        for i in range(min(n_representatives, len(cluster_indices))):
            rep_index = cluster_indices[sorted_indices[i]]
            representatives.append(image_paths[rep_index])

    return representatives

# def select_representatives(features, cluster_labels, image_paths, min_total_representatives=3, max_per_cluster=5):
#     representatives = []
#     for cluster in range(max(cluster_labels) + 1):
#         cluster_indices = np.where(cluster_labels == cluster)[0]
#         cluster_features = features[cluster_indices]
#         cluster_center = np.mean(cluster_features, axis=0)
        
#         distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
#         sorted_indices = np.argsort(distances)
        
#         n_representatives = min(max_per_cluster, len(cluster_indices))
#         for i in range(n_representatives):
#             rep_index = cluster_indices[sorted_indices[i]]
#             representatives.append(image_paths[rep_index])

#     # If we don't have enough representatives, add more from larger clusters
#     while len(representatives) < min_total_representatives:
#         for cluster in range(max(cluster_labels) + 1):
#             if len(representatives) >= min_total_representatives:
#                 break
#             cluster_indices = np.where(cluster_labels == cluster)[0]
#             if len(cluster_indices) > len(representatives) // max(cluster_labels) + 1:
#                 for idx in cluster_indices:
#                     if image_paths[idx] not in representatives:
#                         representatives.append(image_paths[idx])
#                         break

#     return representatives


# Main execution

if __name__ == "__main__":
    input_dir = '/HDD/research/clustering/datasets/tenneco_outer/crop'
    output_dir = '/HDD/research/clustering/datasets/tenneco_outer/outputs_v2'

    offset = 'auto'

    input_dir = osp.join(input_dir, f'offset_{offset}')

    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
    model.eval()

    # Extract features
    features, image_paths = preprocess_and_extract_features(input_dir, model)

    # Apply PCA
    reduced_features = apply_pca(features)

    # Perform clustering
    cluster_labels = perform_kmeans(reduced_features)

    # Select representatives
    representatives = select_representatives(reduced_features, cluster_labels, image_paths)

    # Copy representative images to output folder
    os.makedirs(output_dir, exist_ok=True)
    for rep_path in representatives:
        filename = os.path.basename(rep_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cv2.imread(rep_path))

    print(f"Selected {len(representatives)} representative images out of {len(image_paths)} total images.")
    print(f"Representative images saved to {output_dir}")
