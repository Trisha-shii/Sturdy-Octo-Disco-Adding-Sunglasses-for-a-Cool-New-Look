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


##   PROGRAM :

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread(r"C:\Users\Trisha Priyadarshni\Pictures\Camera Roll\WIN_20250327_10_46_32_Pro.jpg")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

![Screenshot 2025-04-10 102823](https://github.com/user-attachments/assets/41fa657e-653b-4c58-8c34-220cd98a454b)


#resized_faceImage.shape
faceImage.shape

![Screenshot 2025-04-10 102910](https://github.com/user-attachments/assets/eda83a3e-0e26-4e08-8f40-ee2c29b7f807)


# Load the Sunglass image with Alpha channel
# (https://pngtree.com/freepng/red-triangle-sunglasses-black-glass_7296611.html)
glassPNG = cv2.imread(r"C:\Users\Trisha Priyadarshni\Downloads\Sunglass.png",-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

![Screenshot 2025-04-10 102927](https://github.com/user-attachments/assets/a581cd0b-c795-4d75-a7ec-8a45e9aca934)

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
![Screenshot 2025-04-10 102950](https://github.com/user-attachments/assets/54c05fb6-6d87-45db-ad45-d5e1199522a4)



# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');

![Screenshot 2025-04-10 103046](https://github.com/user-attachments/assets/a2697d58-818a-42af-91c4-7fc037504443)


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

![Screenshot 2025-04-10 103126](https://github.com/user-attachments/assets/c7451b13-0c09-45f7-a202-7a3ffa59ada7)


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

![Screenshot 2025-04-10 103146](https://github.com/user-attachments/assets/c5531eee-6a41-41a2-835f-df27f854ade8)



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

![Screenshot 2025-04-10 103200](https://github.com/user-attachments/assets/a7f695ae-e56a-4609-8e25-27eb0146b691)










