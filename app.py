import gradio as gr
from ultralytics import YOLO

model = YOLO("weights/yolov8n_rice_pest.pt")


def detect(image, conf):
    if image is None:
        return None, []
    r = model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    rows = [[r.names[int(c)], round(float(s), 3)] for c, s in zip(r.boxes.cls, r.boxes.conf)]
    return r.plot()[..., ::-1], rows


demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="pil", label="Upload a rice pest photo / Unggah foto hama padi"),
            gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")],
    outputs=[gr.Image(label="Detections / Hasil deteksi"),
             gr.Dataframe(headers=["Pest / Hama", "Confidence"], label="Detected objects")],
    title="Rice Pest Detection with YOLOv8",
    description="Detects 14 rice pest species (IP102 dataset). By Desi Susanti, S.Kom., M.Kom.",
)

if __name__ == "__main__":
    demo.launch()
