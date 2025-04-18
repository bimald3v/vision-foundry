from ultralytics import YOLO
import cv2
import numpy as np

class WarehouseObjectDetector:
    def __init__(self, confidence_threshold=0.3):
        # Load YOLOv8 model
        self.detector = YOLO('yolov8x.pt')
        self.confidence_threshold = confidence_threshold
        
        # Map COCO classes to warehouse-specific labels
        self.class_mapping = {
            'person': 'warehouse worker',
            'truck': 'forklift',
            'car': 'forklift',  # Sometimes forklifts are detected as cars
            'backpack': 'warehouse equipment',
            'handbag': 'warehouse equipment',
            'suitcase': 'cardboard box',
            'sports ball': 'warehouse equipment',
            'box': 'cardboard box',
            'tv': 'pallet rack',  # Large rectangular objects might be shelves
            'laptop': 'cardboard box',
            'mouse': 'warehouse equipment',
            'remote': 'warehouse equipment',
            'keyboard': 'warehouse equipment',
            'cell phone': 'warehouse equipment',
            'book': 'cardboard box',
            'clock': 'warehouse equipment',
            'vase': 'cardboard box',
            'scissors': 'warehouse equipment',
            'teddy bear': 'cardboard box',
            'hair drier': 'warehouse equipment',
            'toothbrush': 'warehouse equipment'
        }
        
        self.colors = {
            'warehouse worker': (255, 255, 0),   # Yellow
            'forklift': (0, 0, 255),            # Red
            'warehouse equipment': (128, 255, 255), # Light Cyan
            'cardboard box': (255, 0, 0),        # Blue
            'pallet rack': (255, 0, 255),        # Magenta
            'unknown': (128, 128, 128)           # Gray
        }
    
    def draw_predictions(self, frame, results):
        # Process each detection
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get confidence score
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class name and map to warehouse label
                    cls = int(box.cls[0])
                    orig_label = r.names[cls]
                    label = self.class_mapping.get(orig_label, 'unknown')
                    
                    # Get color for the label
                    color = self.colors.get(label, self.colors['unknown'])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw filled background for text
                    text = f"{label}: {conf:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1-text_height-10), (x1 + text_width, y1), color, -1)
                    
                    # Draw white text
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def detect_objects_in_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set up video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}", end="")
            
            # Get predictions for this frame using YOLO
            results = self.detector.predict(frame, conf=self.confidence_threshold)
            
            # Draw predictions on frame
            annotated_frame = self.draw_predictions(frame.copy(), results)
            
            if output_path:
                out.write(annotated_frame)
            
            # Display results that meet confidence threshold
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= self.confidence_threshold:
                        cls = int(box.cls[0])
                        orig_label = r.names[cls]
                        mapped_label = self.class_mapping.get(orig_label, 'unknown')
                        print(f"\nFrame {frame_count}: Found {mapped_label} (original: {orig_label}) with confidence {conf:.2f}")
        
        print("\nProcessing complete!")
        cap.release()
        if output_path:
            out.release()

def main():
    # Initialize detector with confidence threshold
    detector = WarehouseObjectDetector(confidence_threshold=0.3)
    
    # Process video file
    video_path = "files/plyneer_sample.mp4"
    output_path = "files/output_plyneer_sample.mp4"
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print("Using YOLOv5 model with warehouse-specific class mapping")
    
    # Process video with visualization
    detector.detect_objects_in_video(
        video_path=video_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()