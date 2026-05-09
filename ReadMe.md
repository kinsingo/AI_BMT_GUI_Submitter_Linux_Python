> **Last Updated:** 2026-05-09 (Version 2.8)
---
## 1. Environment
- ISA(Instruction Set Architecture) : AMD64(x86_64)
- OS : Ubuntu 22.04 LTS, 24.04 LTS
- Python Version: **3.8.X ~ 3.13.X supported**

---
## 2. Build System Set-up
**1. Install Packages**
- Open a terminal and run the following commands to install CMake, g++ compiler, Ninja Build System, and EGL Library.
  ```bash
  sudo apt update
  sudo apt install cmake                     # CMake
  sudo apt install build-essential           # GCC, G++, Make
  sudo apt-get install ninja-build           # Ninja
  sudo apt install libgl1 libgl1-mesa-dev libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0  # EGL and OpenGL
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

**3. Hugging Face Authentication (for LLM tasks)**
If you're using Hugging Face models (especially gated models like Llama), you need to authenticate:

1. **Get Hugging Face Token:**

   - Visit https://huggingface.co/settings/tokens
   - Create a new token with "Read" permission
   - Copy the token

2. **Login via CLI:**

   ```bash
   python -m huggingface_hub.commands.huggingface_cli login
   ```

   Enter your token when prompted.

3. **Access Gated Models:**
   - For gated models (e.g., Llama), visit the model page
   - Example: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - Click "Agree and access repository" to accept terms

**Note:** Without authentication, you'll see `hf_token:None` error when loading gated models.

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

    # Load and initialize your model here
    def initialize(self, model_path: str) -> None:

    # return the implemented interface task type. 
    def getInterfaceType(self) -> InterfaceType:

    # Power measurement selection (default: do not measure)
    def getPowerDeviceType(self) -> PowerDeviceType:

    #  Vision tasks: preprocessing & inference
    #  - preprocessVisionData: convert raw image file into model input format
    #  - inferVision: run inference on preprocessed data and return vision model outputs
    #  - dataTransferVision : transfer vision model outputs to BMT result format
    def preprocessVisionData(self, image_path: str) -> VariantType:
    def inferVision(self, data: List[VariantType]) -> model_outputs:
    def dataTransferVision(self, model_outputs) -> List[BMTVisionResult]:

    # LLM tasks: preprocessing & inference
    # - preprocessLLMData: convert raw input into model input format
    # - inferLLM: run inference on preprocessed data and return LLM model outputs
    # - dataTransferLLM : transfer LLM model outputs to BMT result format
    def preprocessLLMData(self, llmData: LLMPreprocessedInput) -> VariantType:
    def inferLLM(self, data: List[VariantType]) -> model_outputs:
    def dataTransferLLM(self, model_outputs) -> List[BMTLLMResult]: 

    # LLM MMLU tasks: first token generation for TTFT measurement
    # - inferFirstToken: generate only the first token (AI-BMT will measure the time internally)
    # - Returns None (we only measure TTFT, don't care about the actual first token output)
    # - Only used for MMLU tasks that require TTFT measurement
    def inferFirstToken(self, preprocessed_data) -> None:
      
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

### 🔗 Additional Support: Remote GUI Guide

   **Notice**: This application is **designed to operate within a GUI environment**. If you are utilizing a **remote server**, setting up **X11 Forwarding** is necessary to access the interface properly.

   For step-by-step instructions, please refer to the link :
   [**How to Use AIBMT GUI via SSH (Guide)**](https://github.com/kinsingo/SNU_BMT_DOCX/blob/main/X11_GUI_Forwarding_Guide.md)
