import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(image_path):
    """Load and preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size according to your model
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return input_batch, image

def load_model(model_path):
    """Load the PyTorch model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    return model, device

def perform_inference(model, image_tensor, device):
    """Run inference on the input image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # If your model returns multiple outputs, adjust accordingly
        if isinstance(output, tuple):
            detections = output[0]
        else:
            detections = output
            
        return detections

def process_detections(detections, confidence_threshold=0.5):
    """Process model output into a format suitable for visualization"""
    # Convert detections to numpy array
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    
    # Assuming detections format: [x1, y1, x2, y2, confidence, class_id]
    # Filter by confidence
    valid_detections = detections[detections[:, 4] > confidence_threshold]
    
    return valid_detections

def draw_boxes(image, detections):
    """Draw bounding boxes on the image"""
    image_np = np.array(image)
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        
        # Convert coordinates to image scale
        height, width = image_np.shape[:2]
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Draw rectangle
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add label
        label = f'Class {int(class_id)}: {confidence:.2f}'
        cv2.putText(image_np, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image_np

def visualize_results(main_image_vis, query_image_vis):
    """Display results side by side"""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Main Image Detections')
    plt.imshow(main_image_vis)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Query Image Detections')
    plt.imshow(query_image_vis)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set paths
    model_path = 'path/to/your/model.pth'
    main_image_path = 'path/to/main/image.jpg'
    query_image_path = 'path/to/query/image.jpg'
    
    # Load model
    model, device = load_model(model_path)
    
    # Process main image
    main_tensor, main_image = load_image(main_image_path)
    main_detections = perform_inference(model, main_tensor, device)
    main_detections = process_detections(main_detections)
    main_vis = draw_boxes(main_image, main_detections)
    
    # Process query image
    query_tensor, query_image = load_image(query_image_path)
    query_detections = perform_inference(model, query_tensor, device)
    query_detections = process_detections(query_detections)
    query_vis = draw_boxes(query_image, query_detections)
    
    # Visualize results
    visualize_results(main_vis, query_vis)

if __name__ == '__main__':
    main() 