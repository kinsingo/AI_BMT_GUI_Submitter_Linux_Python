# AI-BMT Platform — Python Submitter Interface (Added LLM Tasks, Multi-Domain Tasks, and Custom Dataset Evaluation Modes)
**Last Updated:** 2025-09-03 

---
## 1. Environment
- ISA(Instruction Set Architecture) : AMD64(x86_64)
- OS : Ubuntu 22.04 LTS, 24.04 LTS
- Python Version: **3.8.X ~ 3.12.X supported**

---
## 2. Build System Set-up
**1. Install Packages**
- Open a terminal and run the following commands to install CMake, g++ compiler, Ninja Build System, and EGL Library.
  ```bash
  sudo apt update
  sudo apt install cmake                     # CMake
  sudo apt install build-essential           # GCC, G++, Make
  sudo apt-get install ninja-build           # Ninja
  sudo apt install libgl1 libgl1-mesa-dev    # EGL and OpenGL
  sudo apt install unzip                     # unzip
  ```

**2. Verify the Installation**
- You can check the versions of the installed tools by running the following commands. If these commands return version information for each tool, the installation was successful.
  ```bash
  cmake --version
  gcc --version
  ninja --version
  dpkg -l | grep -E 'libgl1|libgl1-mesa-dev'
  ```

---
## 3. Project Description
1. Implement AI_BMT_Interface to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).
2. Various task example codes are provided. Use these example codes as a reference to implement the interface for the AI Processing Unit.

---
## 4. Submitter Development Guide

### Required Interface
submitter **must** subclass `bmt.AI_BMT_Interface` and implement the following methods:
```python
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def initialize(self, model_path: str) -> None:
        # Load and initialize your model here

    def getInterfaceType(self) -> InterfaceType:
        # return the implemented interface task type. 

    def convertToPreprocessedDataForInference(self, image_path: str) -> VariantType:
        # Perform image loading and preprocessing here

    def runInference(self, data: List[VariantType]) -> List[BMTResult]:
        # Perform inference and return results
```

### Optional Interface

submitter can optionally provide hardware/system metadata using:
```python
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def getOptionalData(self) -> Optional_Data:
        data = Optional_Data()
        data.cpu_type = "Intel i7-9750HF"
        data.accelerator_type = "DeepX M1 (NPU)"
        data.submitter = "DeepX"
        data.cpu_core_count = "16"
        data.cpu_ram_capacity = "32GB"
        data.cooling = "Air"
        data.cooling_option = "Active"
        data.cpu_accelerator_interconnect_interface = "PCIe Gen5 x16"
        data.benchmark_model = "ResNet-50"
        data.operating_system = "Windows 10"
        return data
```

## 5. Start BMT
using following commands in `AI_BMT_GUI_Submitter_Linux_Python/` directory.
```bash
export LD_LIBRARY_PATH=$(pwd)/lib
python main.py
```
