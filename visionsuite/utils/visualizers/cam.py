import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def save_activation(module, input, output):
            self.activations = output
        
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)
    
    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_score = output[0, target_class]
        class_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1)
        
        cam = np.maximum(cam, 0)
        cam = cam[0]
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

# 2. 이미지 전처리 함수
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, np.array(image)

# 3. CAM 시각화 함수
def overlay_cam_on_image(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    overlayed = 0.5 * heatmap + 0.5 * img
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    return overlayed

# 4. 실행 코드
def main(img_path):
    # 모델 및 Grad-CAM 설정
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 이미지 로드 및 전처리
    image_tensor, original_img = preprocess_image(img_path)
    
    # 예측 수행
    output = model(image_tensor)
    target_class = torch.argmax(output).item()
    print(f'Predicted Class: {target_class}')
    
    # Grad-CAM 생성
    cam = grad_cam.generate_cam(image_tensor, target_class)
    overlayed_img = overlay_cam_on_image(original_img, cam)
    
    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlayed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    
    plt.show()

# 실행 예제
if __name__ == "__main__":
    img_path = "/HDD/datasets/visionsuite/cam/0016E5_07959.png"  # 실행할 이미지 파일
    main(img_path)