from typing import ClassVar, List, Type
import json
import numpy as np
import tvm
from tvm import relax


class BaseScheduler:
    """
    Base class for defining scheduler behaviors, providing a framework for loading constants,
    and executing steps during inference based on pre-calculated constants.

    Attributes:
        scheduler_label (ClassVar[str]): A unique label identifying the scheduler type.
        timesteps (List[tvm.nd.NDArray]): List of timesteps for the scheduler.
    """

    scheduler_label: ClassVar[str]
    timesteps: List[tvm.nd.NDArray]

    def __init__(self, artifact_path: str, device) -> None:
        self.load_constants(artifact_path, device)

    def load_constants(self, artifact_path: str, device) -> None:
        raise NotImplementedError()

    def step(self, vm: relax.VirtualMachine, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        raise NotImplementedError()


class PNDMScheduler(BaseScheduler):
    """Scheduler for Probabilistic Neural Decision Models (PNDM)."""
    scheduler_label = "pndm"

    def load_constants(self, artifact_path: str, device) -> None:
        """
        Load and process scheduler constants from a JSON file.

        Args:
            artifact_path (str): Path to the JSON file containing constants.
            device: Device on which tensors should be allocated.
        """
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
        """Convert a list of numeric values into a list of TVM NDArrays."""
        return [tvm.nd.array(np.array(item, dtype=dtype), device) for item in data]

    def update_state(self, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        """
        Update and return the current state of the sample based on the model output and counter.

        Args:
            model_output (tvm.nd.NDArray): The current output from the model.
            sample (tvm.nd.NDArray): The current sample.
            counter (int): Step counter to decide the action.

        Returns:
            tvm.nd.NDArray: Updated sample.
        """
        if counter != 1:
            self.ets = self.ets[-3:] + [model_output]
        if counter == 0:
            self.cur_sample = sample
        elif counter == 1:
            sample = self.cur_sample
        return sample

    def step(self, vm: relax.VirtualMachine, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        """
        Execute one scheduling step using the provided virtual machine and data tensors.

        Args:
            vm (relax.VirtualMachine): The virtual machine to execute scheduling functions.
            model_output (tvm.nd.NDArray): The output from the model at the current step.
            sample (tvm.nd.NDArray): The current sample being processed.
            counter (int): The current step number in the scheduling process.

        Returns:
            tvm.nd.NDArray: The result of the scheduling step.
        """
        sample = self.update_state(model_output, sample, counter)
        step_func = getattr(vm, f"pndm_scheduler_step_{min(counter, 4)}")
        return step_func(sample, model_output, self.sample_coeff[counter], self.alpha_diff[counter], self.model_output_denom_coeff[counter], *self.ets)


class DPMSolverMultistepScheduler(BaseScheduler):
    """
    Scheduler for Diffusion Probabilistic Model (DPM) using a multistep approach to adjust output
    based on pre-defined diffusion steps.

    Attributes:
        scheduler_label (str): Identifies this scheduler as 'dpm-solver-multistep'.
    """
    scheduler_label = "dpm-solver-multistep"

    def load_constants(self, artifact_path: str, device) -> None:
        constants = self.read_json_file(f"{artifact_path}/dpm_solver_multistep_scheduler_constants.json")
        self.timesteps = self.process_data(constants["timesteps"], "int32", device)
        self.alpha = self.process_data(constants["alpha"], "float32", device)
        self.sigma = self.process_data(constants["sigma"], "float32", device)
        self.c0 = self.process_data(constants["c0"], "float32", device)
        self.c1 = self.process_data(constants["c1"], "float32", device)
        self.c2 = self.process_data(constants["c2"], "float32", device)
        self.last_model_output = tvm.nd.empty((1, 4, 64, 64), "float32", device)

    @staticmethod
    def read_json_file(file_path: str) -> dict:
        with open(file_path, "r") as file:
            return json.load(file)

    @staticmethod
    def process_data(data: List[float], dtype: str, device) -> List[tvm.nd.NDArray]:
        return [tvm.nd.array(np.array(item, dtype=dtype), device) for item in data]

    def convert_output(self, vm: relax.VirtualMachine, sample: tvm.nd.NDArray, model_output: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        """
        Converts the model output at each step using pre-computed alpha and sigma constants to adjust
        the sample based on the diffusion model.

        Args:
            vm (relax.VirtualMachine): The virtual machine to execute scheduling functions.
            sample (tvm.nd.NDArray): The current sample tensor.
            model_output (tvm.nd.NDArray): Output from the model at the current step.
            counter (int): The current step number in the scheduling process.

        Returns:
            tvm.nd.NDArray: Adjusted model output ready for the next step in the diffusion process.
        """
        return vm["dpm_solver_multistep_scheduler_conversion_output"](sample, model_output, self.alpha[counter], self.sigma[counter])

    def compute_prev_latents(self, vm: relax.VirtualMachine, sample: tvm.nd.NDArray, model_output: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        """
        Compute the previous latent state based on the current and last outputs of the model,
        using calculated coefficients from the diffusion process.

        Args:
            vm (relax.VirtualMachine): The virtual machine to execute scheduling functions.
            sample (tvm.nd.NDArray): The current sample being processed.
            model_output (tvm.nd.NDArray): Current output from the model.
            counter (int): The current step number in the scheduling process.

        Returns:
            tvm.nd.NDArray: The previous latent state adjusted by the computed coefficients.
        """
        return vm["dpm_solver_multistep_scheduler_step"](sample, model_output, self.last_model_output, self.c0[counter], self.c1[counter], self.c2[counter])

    def step(self, vm: relax.VirtualMachine, model_output: tvm.nd.NDArray, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        """
        Execute one step of the diffusion process using the DPM solver multistep scheduler.
        This involves converting the model output and computing the previous latents.

        Args:
            vm (relax.VirtualMachine): The virtual machine to execute scheduling functions.
            model_output (tvm.nd.NDArray): The output from the model at the current step.
            sample (tvm.nd.NDArray): The current sample being processed.
            counter (int): The current step number in the scheduling process.

        Returns:
            tvm.nd.NDArray: The result of the current scheduling step.
        """
        converted_output = self.convert_output(vm, sample, model_output, counter)
        prev_latents = self.compute_prev_latents(vm, sample, converted_output, counter)
        self.last_model_output = converted_output
        return prev_latents

# List of scheduler classes available for inference
schedulers_inference: List[Type[BaseScheduler]] = [DPMSolverMultistepScheduler, PNDMScheduler]