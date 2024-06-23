import numpy as np
import matplotlib.pyplot as plt

# Define the face matrix
face_matrix = np.array([
    [30, 28, 27, 26, 25, 26, 27, 28, 30],
    [28, 20, 19, 18, 17, 18, 19, 20, 28],
    [27, 19, 12, 11, 10, 11, 12, 19, 27],
    [26, 18, 11,  9,  8,  9, 11, 18, 26],
    [25, 17, 10,  8,  7,  8, 10, 17, 25],
    [26, 18, 11,  9,  8,  9, 11, 18, 26],
    [27, 19, 12, 11, 10, 11, 12, 19, 27],
    [28, 20, 19, 18, 17, 18, 19, 20, 28],
    [30, 28, 27, 26, 25, 26, 27, 28, 30]
])

# Create an image with blue squares for the face and red squares for the background
image_colored = np.zeros((9, 9, 3), dtype=np.uint8)

# Define colors
blue = [0, 0, 255]  # Blue color in RGB format
red = [255, 0, 0]   # Red color in RGB format

# Apply colors to the image based on the face matrix
for i in range(face_matrix.shape[0]):
    for j in range(face_matrix.shape[1]):
        if face_matrix[i, j] < 20:  # Assuming face pixels are those with values less than 20
            image_colored[i, j] = blue
        else:
            image_colored[i, j] = red

# Display the image
plt.imshow(image_colored)
plt.title('Human Face with Colored Squares')
plt.axis('off')
plt.show()