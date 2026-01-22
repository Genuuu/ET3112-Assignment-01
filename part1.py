import cv2 as cv
import numpy as np

f = cv.imread('Images/runway.png', cv.IMREAD_GRAYSCALE)
assert f is not None, "Image not found"

gamma_values = [0.5, 2.0]

for g in gamma_values:
    table = np.array([((i / 255.0) ** g) * 255 for i in np.arange(0, 256)]).astype('uint8')

    h = cv.LUT(f, table)
    
    filename = f'runway_gamma_{g}.png'
    cv.imwrite(filename, h)
    print(f"Saved: {filename}")
    
    cv.imshow(f'Gamma {g}', h)
    cv.waitKey(0)

r1 = int(0.2 * 255)
r2 = int(0.8 * 255)

stretch_table = []
for i in range(256):
    if i < r1:
        val = 0
    elif i > r2:
        val = 255
    else:
        val = ((i - r1) / (r2 - r1)) * 255
    stretch_table.append(val)

stretch_table = np.array(stretch_table).astype('uint8')
contrast_stretched = cv.LUT(f, stretch_table)

cv.imwrite('runway_contrast_stretched.png', contrast_stretched)
print("Saved: runway_contrast_stretched.png")

cv.imshow('Contrast Stretched', contrast_stretched)
cv.waitKey(0)
cv.destroyAllWindows()