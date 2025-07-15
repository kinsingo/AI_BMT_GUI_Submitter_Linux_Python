import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import ExecuteGUI, bmt, current_dir

# Define the interface class for Classification using ONNX
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.array(image, dtype=np.float32).reshape(1, 3, 224, 224)

    def runInference(self, preprocessed_data_list):
        results = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_data})
            output_tensor = outputs[0]  # shape: (1, 1000)
            result = bmt.BMTResult()
            result.classProbabilities = output_tensor.flatten().tolist()
            results.append(result)
        return results

if __name__ == "__main__":
    interface = SubmitterImplementation()
    model_path = current_dir / "Model" / "Classification" / "mobilenet_v2_opset10.onnx"
    model_path = model_path.as_posix()
    ExecuteGUI(interface, model_path)