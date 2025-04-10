# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## PROGRAM :

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread(r"C:\Users\Trisha Priyadarshni\Pictures\Camera Roll\WIN_20250327_10_46_32_Pro.jpg")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

![Screenshot 2025-04-10 102823](https://github.com/user-attachments/assets/4513b9c3-63a7-41e7-ada8-caec1dc311e0)

#resized_faceImage.shape
faceImage.shape

![Screenshot 2025-04-10 102910](https://github.com/user-attachments/assets/8d2306c9-b948-4283-a06d-1619ecc44696)

# Load the Sunglass image with Alpha channel
# (https://pngtree.com/freepng/red-triangle-sunglasses-black-glass_7296611.html)
glassPNG = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\Sunglass.png",-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

![Screenshot 2025-04-10 102927](https://github.com/user-attachments/assets/7c58d2f0-41f8-4497-986a-12ada144323e)

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))


![Screenshot 2025-04-10 102950](https://github.com/user-attachments/assets/9081fe49-f7b3-4fcd-b252-baf7ea9c7b06)

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');

![Screenshot 2025-04-10 103046](https://github.com/user-attachments/assets/d6d65e04-bbdb-43c0-8e78-76a7740f9660)


# Make a fresh copy to avoid cumulative overlays
faceWithGlassesNaive = faceImage.copy()

# Resize glasses to make them bigger
target_width = 300  # Increase width
target_height = 200  # Increase height
glassBGR = cv2.resize(glassBGR, (target_width, target_height))

# Overlay position
x1, y1 = 490, 250
x2, y2 = x1 + target_width, y1 + target_height

# Replace the eye region with the bigger sunglass image
faceWithGlassesNaive[y1:y2, x1:x2] = glassBGR

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.title("Face with Bigger Glasses")
plt.axis('off')
plt.show()

![Screenshot 2025-04-10 103126](https://github.com/user-attachments/assets/35f7ec42-aa84-43d2-8045-5ac5cf7cbd86)

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3-channel image for the mask
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = glassMask / 255.0

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Adjust size and position (bigger + moved up)
x, y, w, h = 470, 250, 350, 200 # x = 40 → Shifted left, y = 160 → Moved up, bigger size

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[y:y + h, x:x + w]

# Resize glassMask and glassBGR to match eyeROI size
glassMask = cv2.resize(glassMask, (eyeROI.shape[1], eyeROI.shape[0]))
glassBGR = cv2.resize(glassBGR, (eyeROI.shape[1], eyeROI.shape[0]))

# Use float32 for better precision
maskedEye = cv2.multiply(eyeROI.astype(np.float32), (1 - glassMask.astype(np.float32)))
maskedGlass = cv2.multiply(glassBGR.astype(np.float32), glassMask.astype(np.float32))

# Combine the masked eye and glass regions
eyeRoiFinal = cv2.add(maskedEye, maskedGlass).astype(np.uint8)

# Overlay result back into the face image
faceWithGlassesArithmetic[y:y + h, x:x + w] = eyeRoiFinal

# Display results
plt.figure(figsize=[20,20])
plt.subplot(131); plt.imshow(maskedEye[...,::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[...,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(faceWithGlassesArithmetic[...,::-1]); plt.title("Augmented Face with Sunglasses")
plt.show()

![Screenshot 2025-04-10 103146](https://github.com/user-attachments/assets/b42c24aa-6666-4eb0-871b-2cb4cd393a88)
import cv2

# Resize eyeRoiFinal to match the target shape (160, 320)
resizedEyeRoi = cv2.resize(eyeRoiFinal, (320, 160))  # (width, height)

# Now you can safely assign it
faceWithGlassesArithmetic[150:150 + 160, 45:45 + 320] = resizedEyeRoi

# Display the final result
plt.figure(figsize=[20,20])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("With Sunglasses")
plt.show()

![Screenshot 2025-04-10 103200](https://github.com/user-attachments/assets/7e2d40d5-1ad3-4e83-a9dc-5683f9867ab0)
```
