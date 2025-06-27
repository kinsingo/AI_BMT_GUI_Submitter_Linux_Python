import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import ExecuteGUI, bmt, current_dir

# Define the interface class for ObjectDetection using ONNX
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.session = None
        self.input_name = None
        self.output_name = None
        
    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""  # e.g., "DeepX M1(NPU)"
        optional.submitter = ""         # e.g., "DeepX"
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional
    
    def Initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        return True

    def convertToPreprocessedDataForInference(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))  # (3, 640, 640)
        return image.flatten().tolist()

    def runInference(self, preprocessed_data_list):
        results = []
        for i, preprocessed_data in enumerate(preprocessed_data_list):
            input_tensor = np.array(preprocessed_data, dtype=np.float32).reshape(1, 3, 640, 640)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output_tensor = outputs[0] # shape : (1, 25200, 85) for YOLOv5, (1, 84, 8400) for Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12, (1, 300, 6) for Yolov10
            result = bmt.BMTResult()
            result.objectDetectionResult = output_tensor.flatten().tolist()
            results.append(result)
        return results

if __name__ == "__main__":
    interface = SubmitterImplementation()
    model_path = current_dir / "Model" / "ObjectDetection" / "Yolov5n_opset12.onnx"
    model_path = model_path.as_posix()
    ExecuteGUI(interface, model_path)