from PIL import Image
import numpy as np
import os

imagesDirectory = "./figure"

i = 0
for imageName in os.listdir(imagesDirectory):
    imagePath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagePath)
    ImageArray = np.array(image)
    row = ImageArray.shape[0]
    col = ImageArray.shape[1]
    print(row, col)

    x_left = row
    x_top = col
    x_right = 0
    x_bottom = 0

    """
    Image.crop(left, up, right, below)
    """
    i += 1
    for r in range(row):
        for c in range(col):
            # if ImageArray[row][col][0] < 255 or ImageArray[row][col][0] ==0:
            if ImageArray[r][c][0] < 255 and ImageArray[r][c][0] != 0:
                if x_top > r:
                    x_top = r
                if x_bottom < r:
                    x_bottom = r
                if x_left > c:
                    x_left = c
                if x_right < c:
                    x_right = c
    print(x_left, x_top, x_right, x_bottom)
    # image = Image.open(imagePath)
    cropped = image.crop((x_left - 5, x_top - 5, x_right + 5, x_bottom + 5))  # (left, upper, right, lower)
    cropped.save("./figure/{}.png".format(imageName[:-4], i))
    print(f"{imageName} completed!")