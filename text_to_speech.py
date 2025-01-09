from gtts import gTTS
import os
import numpy as np

def describe_objects_with_audio_numerical_depth(box_list):
    """
    Describes objects with position and numerical depth using gTTS and plays the audio.

    Parameters:
        box_list (list): List of bounding boxes with tuples (x1, y1, x2, y2, confidence, class_name).
        depth_map (numpy.ndarray): The depth map as a 2D NumPy array.
        img_height (int): Height of the image (default: 480).
        img_width (int): Width of the image (default: 640).
    """
    for bbox in box_list:
        # Classify position
        object_name = bbox[5]
        position = bbox[6]
        mean_depth = bbox[7]
        confidence = int(bbox[4] * 100)
        # Create the description
        object_name = bbox[5]  # class_name
        description = f"There is a {object_name} at the {position}, with a relative depth of {mean_depth:.2f} units. and confidence of {confidence} percent"
        print(description)
        
        # Convert the description to speech
        tts = gTTS(description)
        audio_file = "object_description.mp3"
        tts.save(audio_file)
        
        # Play the audio
        os.system(f"start {audio_file}" if os.name == "nt" else f"mpg123 {audio_file}")
