import cv2
import matplotlib.pyplot as plt
from skimage import color, feature

img = cv2.imread("serval.jpg")
image = color.rgb2gray(img)
hogVec, hogVis = feature.hog(image, visualize=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hogVis)
ax[1].set_title("hog")
plt.show()
