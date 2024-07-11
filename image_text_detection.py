import cv2, easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to image file
image_path = "./images/test1.png"
# image_path = "./images/test2.png"
# image_path = "./images/test3.png"
# image_path = "./images/chinese.jpg"
# image_path = "./images/english.png"


# Read image file
img = cv2.imread(image_path)

# If image does not exist, raise ValueError
if img is None:
    raise ValueError("Error loading the image. Please check the file path.")                     

# easyocr text detector, looking for English text
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)         # Make sure GPU is cuda enabled and setup correctly

# Detect text in image
text = reader.readtext(img)         

threshold = 0.25
# Draws box and text
for t in text:
    print(t)        

    bbox, text, score = t           # List containing x,y coordinates of the box's corners, detected text, and accuracy of detected text (1 = 100% certainty)

    # Only draws box and text on detections if it scores is above 0.25
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

# Saves numpy.ndarray results to an image
results = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(results)
image.save("easyocr_results.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

