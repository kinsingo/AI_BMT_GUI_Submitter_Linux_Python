import os
import numpy as np
import cv2
import onnxruntime as ort
#from ModelLoadingHelper import DeeplabWithUpsample #this may necessary for loading pth models
from GUI_Mananger import ExecuteGUI, bmt
from example.Image_Classification_Implementaion import Classification_Implementation, Classification_CustomDataset_Implementation
from example.Image_Segmentation_Implementation import Segmentation_Implementation, Segmentation_CustomDataset_Implementation
from example.ObjectDetection_Implementation import ObjectDetection_Implementation, ObjectDetection_CustomDataset_Implementation
#from example.LLM_Implementation_HuggingFace_Decoder import LLM_Implementation_HuggingFace_Decoder
#from example.LLM_Implementation_HuggingFace_Bert import LLM_Implementation_HuggingFace_Bert

if __name__ == "__main__":
    # -- For Single Task --
    interface = Classification_Implementation() 
    #interface = Classification_CustomDataset_Implementation() 
    #interface = ObjectDetection_Implementation()
    #interface = ObjectDetection_CustomDataset_Implementation()
    #interface = Segmentation_Implementation() 
    #interface = Segmentation_CustomDataset_Implementation()
    #interface = LLM_Implementation_HuggingFace_Decoder()
    #interface = LLM_Implementation_HuggingFace_Bert()
    ExecuteGUI(interface)

    # -- For Multi-Domain Tasks --
    # interface1 = Classification_Implementation()
    # interface2 = ObjectDetection_Implementation()
    # interface3 = Segmentation_Implementation()
    # ExecuteGUI([interface1, interface2, interface3])
    
