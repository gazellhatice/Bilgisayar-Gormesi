import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")  # YOLOv8 model dosyasının bulunduğu yol

# Belirtilen URL'ler
urls = [
    "https://image.hurimg.com/i/hurriyet/75/750x422/671cac2600693dcb1d7c2d57.jpg",
    "https://image.hurimg.com/i/hurriyet/75/750x422/671d0559855e09ba7cc6ff66.jpg",
    "https://img.internethaber.com/rcman/Cw714h402q95gc/storage/files/images/2024/10/26/mehmet-simsek-adhv_cover.jpg",
    "https://img.internethaber.com/rcman/Cw714h402q95gc/storage/files/images/2024/10/26/abdli-sanatci-beyonce-abd-baskan-bhfm_cover.jpg",
    "https://image.milimaj.com/i/milliyet/75/869x477/671ce0ef0b1b6cde247c3f5f.jpg",
    "https://image.hurimg.com/i/hurriyet/75/866x494/662cd59c6d75efde944b841d.jpg",
]

# Her URL için görüntüyü indir ve nesne tespiti yap
for url in urls:
    # Görüntüyü indir
    resp = urllib.request.urlopen(url)
    image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Resmi tahmin et
    results = model(image)

    # Sonuçları işleme
    for result in results:
        # Tahmin edilen sınıflar
        for box in result.boxes:
            class_id = int(box.cls)  # Sınıf ID'si
            confidence = box.conf.item()  # Güven oranını float'a çevir
            label = model.names[class_id]  # Sınıf adı

            if confidence > 0.5:  # Güven eşik değeri
                # Nesne konumunu al
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tespit edilen kutunun koordinatları

                # Nesne etiketini ve güven oranını yazdır
                text = f"{label} {confidence:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Tespit edilen kutuyu çiz
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Metni yaz

    # Sonuçları görüntüle
    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
