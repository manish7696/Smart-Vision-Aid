#python3.10
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the function for transforming the input image
def transform_image(img, model_type="DPT_Large"):
    midas_transforms = torch.hub.load('intel-isl/Midas', 'transforms')
    transform = midas_transforms.dpt_transform if model_type in ['DPT_Large', 'DPT_Hybrid'] else midas_transforms.small_transform
    return transform(img)

# Define the function for performing inference
def infer_depth(input_batch, model, device, img_shape):
    # Pass the image through the model and resize output to input dimensions
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    return prediction.cpu().numpy()

def mean_depth_in_bbox(bbox, depth_map):
    """
    Calculates the mean pixel value of the depth map within a bounding box.

    Parameters:
        bbox (tuple): A tuple containing (x1, y1, x2, y2, confidence, class_name).
        depth_map (numpy.ndarray): The depth map as a 2D NumPy array.

    Returns:
        float: The mean depth value within the bounding box.
    """
    x1, y1, x2, y2, confidence, class_name = bbox

    # Ensure bounding box coordinates are within the depth map bounds
    # x1 = max(0, int(x1))
    # y1 = max(0, int(y1))
    # x2 = min(depth_map.shape[1] - 1, int(x2))
    # y2 = min(depth_map.shape[0] - 1, int(y2))
    # print(depth_map.shape)

    # Extract the region within the bounding box
    bbox_region = depth_map[y1:y2, x1:x2]

    # Calculate the mean depth value
    if bbox_region.size > 0:
        mean_depth = np.mean(bbox_region)
    else:
        mean_depth = 0.0  # Handle empty bounding boxes

    return mean_depth

# Main code
if __name__ == "__main__":
    filename = 'test_image.jpg'
    img = cv2.imread(filename)
    if img is None:
        raise ValueError("Image not found. Check the file path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display input image
    # plt.imshow(img)
    # plt.title("Input Image")
    # plt.axis('off')
    # plt.show()

    # Load MiDaS model and set device
    model_type = "DPT_Large"
    midas = torch.hub.load('intel-isl/Midas', model_type)
    device = torch.device("cuda")
    print(device)
    midas.to(device)
    midas.eval()

    # Transform the image
    input_batch = transform_image(img, model_type).to(device)

    # Perform inference
    depth_map = infer_depth(input_batch, midas, device, img.shape)

    # Display depth map
    plt.imshow(depth_map, cmap='viridis')
    plt.title("Depth Map")
    plt.axis('off')
    plt.show()
