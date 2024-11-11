import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile projeyi indir
rf = Roboflow(api_key="rwHgQjCTFtGDTt9IjcEM")
project = rf.workspace("trafiklambasi-cgheh").project("kayisi-gftvp")
version = project.version(1)
dataset = version.download("yolov8")

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
if __name__ == "__main__":
    # Eğitim işlemi
    results = model.train(data=f"{dataset.location}/data.yaml", epochs=2, batch=8)  # batch size'ı azalt

    # Eğitim sonuçlarının kaydedildiği yolu yazdır
    print(f"Model saved in: {results.save_dir}")  # Kaydedilen modelin bulunduğu dizin
