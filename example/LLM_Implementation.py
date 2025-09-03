import os
import numpy as np
import onnxruntime as ort
from GUI_Mananger import bmt

# LLM용 Submitter (ONNX Runtime)
class LLM_Implementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.session: ort.InferenceSession | None = None
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.modelHasTokenType: bool = False
        self.modelHasAttnMask: bool = False

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""   
        optional.submitter = ""          
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""   # e.g., "32GB"
        optional.cooling = ""            # e.g., "Air"
        optional.cooling_option = ""     # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional
    
    def getInterfaceType(self):
        return bmt.InterfaceType.LLM

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_names  = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.modelHasTokenType = ("token_type_ids"  in self.input_names)
        self.modelHasAttnMask  = ("attention_mask" in self.input_names)
        return True

    def preprocessLLMData(self, llmData: bmt.LLMPreprocessedInput):
        """
        llmData: bmt.LLMPreprocessedInput (fields: input_ids, attention_mask, token_type_ids)
        """
        S = len(llmData.input_ids)
        if self.modelHasAttnMask:
            if len(llmData.attention_mask) != S:
                llmData.attention_mask = [1] * S
        if self.modelHasTokenType:
            if len(llmData.token_type_ids) != S:
                llmData.token_type_ids = [0] * S
        return llmData

    def inferLLM(self, preprocessed_data_list):
        results: list[bmt.BMTLLMResult] = []
        for preprocessed_data in preprocessed_data_list:
            S = len(preprocessed_data.input_ids)
            shape = (1, S)

            feed: dict[str, np.ndarray] = {}
            for nm in self.input_names:
                if nm == "input_ids":
                    feed[nm] = np.asarray(preprocessed_data.input_ids, dtype=np.int64).reshape(shape)
                elif nm == "attention_mask" and self.modelHasAttnMask:
                    feed[nm] = np.asarray(preprocessed_data.attention_mask, dtype=np.int64).reshape(shape)
                elif nm == "token_type_ids" and self.modelHasTokenType:
                    feed[nm] = np.asarray(preprocessed_data.token_type_ids, dtype=np.int64).reshape(shape)
                else:
                    pass

            # Fill the result structure
            outs = self.session.run(self.output_names if self.output_names else None, feed)
            r = bmt.BMTLLMResult()
            r.rawOutputShape = [int(x) for x in outs[0].shape]
            r.rawOutput = np.asarray(outs[0], dtype=np.float32).ravel().tolist()
            results.append(r)
        return results
