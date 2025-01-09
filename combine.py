# from capture import transform_image, infer_depth
# from object_detect import load_model, run_inference

# #python3.10
# import cv2
# import torch
# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     # input image 
#     filename = 'test_image.jpg'
#     img = cv2.imread(filename)
#     if img is None:
#         raise ValueError("Image not found. Check the file path.")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#     # model for depth estimation
#     model_type = "DPT_Large"
#     midas = torch.hub.load('intel-isl/Midas', model_type)
#     device = torch.device("cuda")
#     print(device)
#     midas.to(device)
#     midas.eval()

#     # Transform the image for depth estimation
#     input_batch = transform_image(img, model_type).to(device)

#     # Perform inference for depth estimation
#     depth_map = infer_depth(input_batch, midas, device, img.shape)

#     plt.imshow(depth_map, cmap='viridis')
#     plt.title("Depth Map")
#     plt.axis('off')
#     plt.show()


from capture import transform_image, infer_depth, mean_depth_in_bbox
from object_detect import load_model, run_inference, process_results, classify_bbox_position
from text_to_speech import describe_objects_with_audio_numerical_depth

import cv2
import torch
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

if __name__ == "__main__":

    # Initialize the video capture from webcam or video file
    video_source = 0  # Set to 0 for webcam or provide a video file path
    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        raise ValueError("Video source not accessible. Check the file path or camera connection.")

    # Set desired FPS
    
    frame_interval = 12

    # Load depth estimation model
    model_type = "DPT_Large"
    midas = torch.hub.load('intel-isl/Midas', model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    midas.to(device)
    midas.eval()

    # model for object detection
    model_path = "yolo11n.pt"  # Path to the YOLO model
    model_obj_detect = YOLO(model_path)
    classes = model_obj_detect.names
    print("Press 'q' to quit.")

    try:
        while True:
            start_time = time.time()

            # Read the next frame from the video
            ret, frame = capture.read()
            if not ret:
                print("End of video or failed to capture frame.")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform the frame for depth estimation
            input_batch = transform_image(frame_rgb, model_type).to(device)

            # Perform inference for depth estimation
            depth_map = infer_depth(input_batch, midas, device, frame.shape)
            # print(depth_map.shape)

            obj_inference = model_obj_detect(frame_rgb)
            box_list = []
            for obj in obj_inference:
                boxes = obj.boxes # bounding boxes around the object
                # print(boxes)
                
            
                for box_idx, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])  # Convert coordinates to integers
                    confidence = boxes.conf[box_idx]  # Confidence score for this box
                    confidence = float(confidence.item())
                    class_idx = int(boxes.cls[box_idx])  # Class index for this box
                    class_name = classes[class_idx]  # Get class name from model
                    if confidence > 0.5:
                        box_list.append([x1, y1, x2, y2, confidence, class_name])
                
       
            for i in range(len(box_list)):
                pos = classify_bbox_position(box_list[i])
                depth = mean_depth_in_bbox(box_list[i], depth_map)

                box_list[i].extend([pos, depth])
                
            print(box_list)
            describe_objects_with_audio_numerical_depth(box_list)
            
            # Display the depth map
            # plt.imshow(depth_map, cmap='viridis')
            # plt.title("Depth Map")
            # plt.axis('off')
            # plt.pause(0.001)  # Pause to update the plot

            # Optional: Save the depth map as an image (e.g., PNG)
            # output_filename = f"depth_frame_{int(capture.get(cv2.CAP_PROP_POS_FRAMES))}.png"
            # plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

            # Wait for the appropriate time to maintain 10 FPS
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_interval:
                time.sleep(frame_interval - elapsed_time)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the video capture and close any open plots
        capture.release()
        plt.close()
        print("Video capture released and program terminated.")
