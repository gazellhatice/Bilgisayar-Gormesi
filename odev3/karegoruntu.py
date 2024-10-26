import numpy as np
import cv2
from matplotlib import pyplot as plt

# Kare bir görüntü simüle etme (örnek bir 100x100 kare siyah-beyaz görüntü)
img = np.zeros((100, 100), dtype="uint8")  # "uint8" olarak güncelledik
cv2.rectangle(img, (10, 10), (90, 90), 255, -1)  # Beyaz bir kare çiz

# Kenar tespiti için Canny edge detector kullan
edges = cv2.Canny(img, 100, 200)

# Görüntü ve kenarları göster
plt.subplot(121)
plt.imshow(img, cmap="gray")  # cmap argümanını "gray" olarak ayarla
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap="gray")  # cmap argümanını "gray" olarak ayarla
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()
