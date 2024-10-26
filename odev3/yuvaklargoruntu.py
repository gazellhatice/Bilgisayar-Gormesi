import numpy as np
import cv2
from matplotlib import pyplot as plt

# Yuvarlak bir görüntü simüle etme (örnek bir 100x100 siyah-beyaz görüntü)
img = np.zeros((100, 100), dtype="uint8")
cv2.circle(img, (50, 50), 40, 255, -1)  # Beyaz bir daire çiz

# Kenar tespiti için Canny edge detector kullan
edges = cv2.Canny(img, 100, 200)

# Görüntü ve kenarları göster
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap="gray")
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()
