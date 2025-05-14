import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer

def load_model(config_file, checkpoint_file, device='cuda:0'):
    """
    Load the model with the given configuration and checkpoint.
    
    Args:
        config_file (str): Path to the config file
        checkpoint_file (str): Path to the checkpoint file
        device (str): Device to run the model on
    
    Returns:
        model: Loaded model
    """
    model = init_detector(config_file, checkpoint_file, device=device)
    return model

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        image: Preprocessed image
        original_image: Original image for visualization
    """
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = original_image.copy()
    return image, original_image

def visualize_detections(image, result, score_threshold=0.3):
    """
    Visualize the detection results.
    
    Args:
        image: Original image
        result: Detection results from model
        score_threshold (float): Threshold for showing detections
    """
    # Initialize visualizer
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {
        'classes': ('your_class_names_here',),  # Update with your class names
        'palette': [(255, 0, 0)]  # Update with your color scheme
    }
    
    # Draw detections
    visualizer.add_datasample(
        name='',
        image=image,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=score_threshold
    )
    
    # Get the visualization result
    vis_image = visualizer.get_image()
    
    return vis_image

def main(main_image_path, query_image_path, config_file, checkpoint_file, score_threshold=0.3):
    """
    Main inference function.
    
    Args:
        main_image_path (str): Path to the main image
        query_image_path (str): Path to the query image
        config_file (str): Path to model config file
        checkpoint_file (str): Path to model checkpoint file
        score_threshold (float): Threshold for showing detections
    """
    # Load model
    model = load_model(config_file, checkpoint_file)
    
    # Process main image
    main_image, main_original = preprocess_image(main_image_path)
    main_result = inference_detector(model, main_image)
    
    # Process query image
    query_image, query_original = preprocess_image(query_image_path)
    query_result = inference_detector(model, query_image)
    
    # Visualize results
    main_vis = visualize_detections(main_original, main_result, score_threshold)
    query_vis = visualize_detections(query_original, query_result, score_threshold)
    
    # Create figure for visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Main Image Detections')
    plt.imshow(main_vis)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Query Image Detections')
    plt.imshow(query_vis)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    config_file = 'path/to/your/config.py'
    checkpoint_file = 'path/to/your/checkpoint.pth'
    main_image_path = 'path/to/main/image.jpg'
    query_image_path = 'path/to/query/image.jpg'
    
    main(main_image_path, query_image_path, config_file, checkpoint_file, score_threshold=0.3) 