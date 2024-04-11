import json
from typing import Callable, ClassVar, Dict, List, Type

import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R


class BaseScheduler:
    constants_file_name: ClassVar[str]

    @staticmethod
    def scheduler_steps() -> tvm.IRModule:
        raise NotImplementedError()
    
    @staticmethod
    def list_step_functions() -> List[str]:
        raise NotImplementedError()
    
    @staticmethod
    def calculate_constants() -> Dict[str, List[tvm.nd.NDArray]]:
        raise NotImplementedError()
    

schedulers_build: List[Type[BaseScheduler]]


def calculate_pndm_constants(num_train_timesteps: int, num_inference_steps: int, steps_offset: int, beta_start: float, beta_end: float) -> Dict[str, List[tvm.nd.NDArray]]:
    betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype="float32") ** 2
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    final_alpha_cumprod = alphas_cumprod[0]
    
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = np.arange(0, num_inference_steps) * step_ratio
    timesteps = np.round(timesteps).astype(int) + steps_offset
    timesteps = np.concatenate([timesteps[:-1], timesteps[-2:-1], timesteps[-1:]])[::-1]
    
    sample_coeffs, alpha_diffs, model_output_denom_coeffs = [], [], []
    
    for idx, timestep in enumerate(timesteps):
        prev_timestep = timestep - step_ratio
        
        if idx == 1:
            prev_timestep = timestep
            timestep += step_ratio
        
        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        alpha_diff = alpha_prod_t_prev - alpha_prod_t
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** 0.5 + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev) ** 0.5
        
        sample_coeffs.append(sample_coeff.item())
        alpha_diffs.append(alpha_diff.item()) 
        model_output_denom_coeffs.append(model_output_denom_coeff.item())
    
    const_dict = {}
    const_dict["num_steps"] = len(timesteps)
    const_dict["timesteps"] = timesteps.tolist()
    const_dict["sample_coeff"] = sample_coeffs
    const_dict["alpha_diff"] = alpha_diffs
    const_dict["model_output_denom_coeff"] = model_output_denom_coeffs
    
    return const_dict


def get_pndm_step_functions() -> List[Callable]:
    def step1(mo, *args):
        return mo
    
    def step2(mo, *args):
        return (mo + args[-1]) / relax.const(2, "float32")
    
    def step3(mo, *args):   
        return (relax.const(3, "float32") * args[-1] - args[-2]) / relax.const(2, "float32")
    
    def step4(mo, *args):
        return (relax.const(23, "float32") * args[-1] - relax.const(16, "float32") * args[-2] + relax.const(5, "float32") * args[-3]) / relax.const(12, "float32") 
    
    def step5(mo, *args):
        return relax.const(1 / 24, "float32") * (relax.const(55, "float32") * args[-1] - relax.const(59, "float32") * args[-2] + relax.const(37, "float32") * args[-3] - relax.const(9, "float32") * args[-4])
    
    return [step1, step2, step3, step4, step5]


def pndm_scheduler_step_wrapper(f_output: Callable):
    def scheduler_step(sample, model_output, sample_coeff, alpha_diff, model_output_denom_coeff, ets0, ets1, ets2, ets3):
        output = f_output(model_output, ets0, ets1, ets2, ets3)
        return compute_previous_sample(sample, output, sample_coeff, alpha_diff, model_output_denom_coeff)
    
    return scheduler_step
        
        
def compute_previous_sample(sample, model_output, sample_coeff, alpha_diff, model_output_denom_coeff):
    return sample_coeff * sample - alpha_diff * model_output / model_output_denom_coeff
        
        
def construct_pndm_step_modules(scheduler_step_functions: List[Callable]) -> tvm.IRModule:
    def emit_step(bb, scheduler_step, idx):
        sample = relax.Var("sample", R.Tensor((1, 4, 64, 64), "float32"))
        model_output = relax.Var("model_output", R.Tensor((1, 4, 64, 64), "float32")) 
        sample_coeff = relax.Var("sample_coeff", R.Tensor((), "float32"))
        alpha_diff = relax.Var("alpha_diff", R.Tensor((), "float32"))
        model_output_denom_coeff = relax.Var("model_output_denom_coeff", R.Tensor((), "float32"))
        ets = [relax.Var(f"ets{i}", R.Tensor((1, 4, 64, 64), "float32")) for i in range(4)]
        
        with bb.function(f"pndm_scheduler_step_{idx}", [sample, model_output, sample_coeff, alpha_diff, model_output_denom_coeff, *ets]):
            prev_sample = bb.emit(scheduler_step(sample, model_output, sample_coeff, alpha_diff, model_output_denom_coeff, *ets), "prev_sample")
            bb.emit_func_output(prev_sample)
    
    bb = relax.BlockBuilder()
    
    for idx, scheduler_step in enumerate(scheduler_step_functions):
        scheduler_step = pndm_scheduler_step_wrapper(scheduler_step)  
        emit_step(bb, scheduler_step, idx)
        
    return bb.get()
        

class PNDMScheduler(BaseScheduler):
    constants_file_name = "pndm_scheduler_constants.json"

    @staticmethod
    def scheduler_steps() -> tvm.IRModule:
        scheduler_step_functions = get_pndm_step_functions()
        return construct_pndm_step_modules(scheduler_step_functions)
    
    @staticmethod 
    def list_step_functions() -> List[str]:
        return [f"pndm_scheduler_step_{i}" for i in range(5)]
    
    @staticmethod
    def calculate_constants() -> Dict[str, List[tvm.nd.NDArray]]:
        return calculate_pndm_constants(1000, 50, 1, 0.00085, 0.012)