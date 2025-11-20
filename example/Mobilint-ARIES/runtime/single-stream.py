import os
import numpy as np
import cv2
import onnxruntime as ort
#from ModelLoadingHelper import DeeplabWithUpsample #this may necessary for loading pth models
from GUI_Mananger import ExecuteGUI, bmt
import maccel
from PIL import Image
import torchvision.transforms.functional as F

# Define the interface class for Classification using ONNX
class Classification_Implementation(bmt.AI_BMT_Interface):
    def __init__(self, isGlobalMode, use_customDataset):
        super().__init__()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.isGlobalMode = isGlobalMode
        self.use_customDataset = use_customDataset
        self.acc = maccel.Accelerator()
        self.mc = maccel.ModelConfig()
        if self.isGlobalMode:
            print("global 8 core mode")
            #mc.set_global8_core_mode()
            self.mc.set_global_core_mode({
                maccel.Cluster.Cluster0,maccel.Cluster.Cluster1
            })
        else:
            print("single core mode")
            self.mc.set_single_core_mode(core_ids = [
                maccel.CoreId(maccel.Cluster.Cluster0, maccel.Core.Core0),
            ])

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = "mobilint(python) global" if self.isGlobalMode else "mobilint(python) single"
        optional.submitter = ""         
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional

    def getInterfaceType(self):
        if self.use_customDataset:
            return bmt.InterfaceType.ImageClassification_CustomDataset
        else:
            return bmt.InterfaceType.ImageClassification

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if self.session != None:
            self.session.dispose()

        model = maccel.Model(model_path,self.mc)
        model.launch(self.acc)
        self.session=model
        return True

    def preprocessVisionData(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         # Apply custom dataset preprocessing if needed
        if self.use_customDataset:
            image = Image.fromarray(image)
            image = F.resize(image, 232)
            image = F.center_crop(image, [224, 224])
            image = np.array(image)

        image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.array(image, dtype=np.float32).reshape(1, 3, 224, 224)
    
    def inferVision(self, preprocessed_data_list):
        output_tensors = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs= self.session.infer([preprocessed_data])[0]
            output_tensors.append(outputs[0])
        return output_tensors
    
    def dataTransferVision(self, output_tensors):
        results = []
        for output_tensor in output_tensors:
            result = bmt.BMTVisionResult()
            result.classProbabilities = output_tensor.flatten()
            results.append(result)
        return results

if __name__ == "__main__":
    isGlobalMode = False
    use_customDataset = False
    interface = Classification_Implementation(isGlobalMode, use_customDataset)
    ExecuteGUI(interface)
    
