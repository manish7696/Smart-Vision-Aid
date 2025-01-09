from ultralytics import YOLO
import cv2

def load_model(model_path):
    """
    Load a YOLO model from the given path.

    Args:
        model_path (str): Path to the YOLO model file.

    Returns:
        YOLO: Loaded YOLO model.
    """
    return YOLO(model_path)


def run_inference(model, image_paths):
    """
    Run inference on a list of images using the YOLO model.

    Args:
        model (YOLO): Loaded YOLO model.
        image_paths (list): List of image file paths.

    Returns:
        list: List of YOLO detection results.
    """
    return model(image_paths)


def process_results(results, image, classes):
    """
    Process and display YOLO inference results.

    Args:
        results (list): List of YOLO results objects.
        save_dir (str): Directory to save processed result images.
    """
    for idx, result in enumerate(results):
        print(f"Processing result {idx + 1}/{len(results)}")
        boxes = result.boxes  # Bounding box outputs
        masks = result.masks  # Segmentation mask outputs (if applicable)
        keypoints = result.keypoints  # Keypoints for pose estimation
        probs = result.probs  # Probabilities for classification
        obb = result.obb  # Oriented bounding box outputs

        print(f"Bounding Boxes:\n{boxes}")
        if masks is not None:
            print(f"Segmentation Masks:\n{masks}")
        if keypoints is not None:
            print(f"Keypoints:\n{keypoints}")
        if probs is not None:
            print(f"Classification Probabilities:\n{probs}")
        if obb is not None:
            print(f"Oriented Bounding Boxes:\n{obb}")

        box_list = []
        # for box_idx, box in enumerate(boxes.xyxy):
        #         x1, y1, x2, y2 = map(int, box)  # Convert to integers for indexing
        #         box_list.append((x1, y1,x2, y2))
        #         confidence = boxes.conf[box_idx]  # Confidence score for this box
        #         box_list.append((x1, y1, x2, y2, confidence))
        #         print(f"Box {box_idx + 1}: {x1, y1, x2, y2}, Confidence: {confidence:.2f}")
        
        #         # cropped_region = image[y1:y2, x1:x2]
        #         # print(cropped_region, "\n|||||||||||\n")
        #         # cv2.imshow('_', cropped_region)
        if boxes is not None:
            for box_idx, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])  # Convert coordinates to integers
                confidence = boxes.conf[box_idx]  # Confidence score for this box
                confidence = float(confidence.item())
                class_idx = int(boxes.cls[box_idx])  # Class index for this box
                class_name = classes[class_idx]  # Get class name from model
                if confidence > 0.8:
                    box_list.append((x1, y1, x2, y2, confidence, class_name))
                # print(box)
                print(f"Box {box_idx + 1}: {x1, y1, x2, y2}, Confidence: {confidence:.2f}, class name:{class_name}")
        
        
        

        # cv2.imshow('obj', cropped_region)

        # Display and save the result
        # result.show()    # for showing object detection result 
        return box_list
          # Save results

def classify_bbox_position(bbox, img_height=480, img_width=640):
    """
    Classifies the position of a bounding box in a 3x3 grid.

    Parameters:
        bbox (tuple): A tuple containing (x1, y1, x2, y2, confidence, class_name).
        img_height (int): Height of the image (default: 480).
        img_width (int): Width of the image (default: 640).

    Returns:
        str: Position of the object in the 3x3 grid (e.g., "top left", "front", "bottom right").
    """
    x1, y1, x2, y2, confidence, class_name = bbox

    # Calculate the center of the bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Define grid boundaries
    h_third = img_height / 3
    w_third = img_width / 3

    # Determine the position in the 3x3 grid
    if y_center < h_third:
        if x_center < w_third:
            position = "top left"
        elif x_center < 2 * w_third:
            position = "top"
        else:
            position = "top right"
    elif y_center < 2 * h_third:
        if x_center < w_third:
            position = "left"
        elif x_center < 2 * w_third:
            position = "front"
        else:
            position = "right"
    else:
        if x_center < w_third:
            position = "bottom left"
        elif x_center < 2 * w_third:
            position = "bottom"
        else:
            position = "bottom right"

    return position


def main():
    # Load the model
    model_path = "yolo11n.pt"  # Path to the YOLO model
    model = load_model(model_path)
    classes = model.names
    # List of images for inference
    image_paths = ["test_image.jpg"]
    image = cv2.imread(image_paths[0])
    # print(image)
    # Run inference
    results = run_inference(model, image_paths)


    # Process and visualize results
    box_list = process_results(results, image, classes)
    print(box_list)


if __name__ == "__main__":
    main()