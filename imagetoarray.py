#pip3 install Pillow

from PIL import Image
import numpy as np



# Load the image
img = Image.open('/home/shiam/Desktop/Deepfake_figures/man_1.jpg')

# Convert the image to greyscale
img_grey = img.convert('L')

# Convert the image data to an array
img_array = np.array(img_grey)

# Print the array of greyscale values
print(img_array)

# Save the greyscale values to a text file
np.savetxt('output_greyscale.txt', img_array, fmt='%d')

# Ensure the image is in RGB mode
if img.mode != 'RGB':
    img = img.convert('RGB')

# Convert the image data to an array
img_array = np.array(img)

# Save the RGB values to a text conversion error
# You used 'img_game' which is not defined; it should be 'img_array'
np.savetxt('output_rgb_values.txt', img_array.reshape(-1, 3), fmt='%d')