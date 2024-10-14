import cv2
import torch

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_and_draw(video_path, output_path):
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    # Параметры для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Выполнение детекции на текущем кадре
        results = model(frame)
        
        # Отрисовка результатов детекции
        for obj in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = obj[:6]
            label = f'{model.names[int(cls)]} {conf:.2f}'
            if model.names[int(cls)] == 'person':
                # Отрисовка рамки вокруг объекта и метки
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Сохранение текущего кадра в выходной файл
        out.write(frame)
    
    # Закрытие всех окон
    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = 'crowd.mp4'
    output_path = 'output_crowd.mp4'
    detect_and_draw(video_path, output_path)
