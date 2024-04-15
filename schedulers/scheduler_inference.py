from typing import ClassVar, List, Type
import json
import numpy as np
import tvm
from tvm import relax


class BaseScheduler:
    scheduler_label: ClassVar[str]
    timesteps: List[tvm.nd.NDArray]

    def __init__(self, artifact_path: str, device) -> None:
        self.load_constants(artifact_path, device)

    def load_constants(self, artifact_path: str, device) -> None:
        raise NotImplementedError()

    def step(self, vm: relax.VirtualMachine, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        raise NotImplementedError()


class PNDMScheduler(BaseScheduler):
    scheduler_label = "pndm"

    def load_constants(self, artifact_path: str, device) -> None:
        constants = self.read_json_file(f"{artifact_path}/pndm_scheduler_constants.json")
        self.timesteps = self.process_data(constants["timesteps"], "int32", device)
        self.sample_coeff = self.process_data(constants["sample_coeff"], "float32", device)
        self.alpha_diff = self.process_data(constants["alpha_diff"], "float32", device)
        self.model_output_denom_coeff = self.process_data(constants["model_output_denom_coeff"], "float32", device)
        self.ets = [tvm.nd.empty((1, 4, 64, 64), "float32", device)] * 4
        self.cur_sample = None

    @staticmethod
    def read_json_file(file_path: str) -> dict:
        with open(file_path, "r") as file:
            return json.load(file)

    @staticmethod
    def process_data(data: List[float], dtype: str, device) -> List[tvm.nd.NDArray]:
        return [tvm.nd.array(np.array(item, dtype=dtype), device) for item in data]

    def update_state(self, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        if counter != 1:
            self.ets = self.ets[-3:] + [model_output]
        if counter == 0:
            self.cur_sample = sample
        elif counter == 1:
            sample = self.cur_sample
        return sample

    def step(self, vm: relax.VirtualMachine, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        sample = self.update_state(model_output, sample, counter)
        step_func = getattr(vm, f"pndm_scheduler_step_{min(counter, 4)}")
        return step_func(sample, model_output, self.sample_coeff[counter], self.alpha_diff[counter], self.model_output_denom_coeff[counter], *self.ets)
