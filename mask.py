import cv2

# LOAD MASK
img_path = 'img.jpg'
img = cv2.imread(img_path, 0)
img_grid = img / 255.0 #sets black to 0 and white to 1 

img_grid = np.transpose(img_grid)

# CREATE GRID
print("Image is: ", img)

print("Image is: ", img_grid)
