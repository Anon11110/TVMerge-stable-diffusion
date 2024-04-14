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
        

def compute_dpm_solver_multistep_scheduler_consts(num_train_timesteps: int, num_inference_steps: int, beta_start: float, beta_end: float) -> Dict[str, List[tvm.nd.NDArray]]:
    betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype="float32") ** 2
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alpha_t = np.sqrt(alphas_cumprod)
    sigma_t = np.sqrt(1 - alphas_cumprod)
    lambda_t = np.log(alpha_t) - np.log(sigma_t)
    
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1).round().astype(int)[::-1][:-1]
    
    alphas, sigmas, c0s, c1s, c2s = [], [], [], [], []
    
    for idx, timestep in enumerate(timesteps):
        t = timesteps[idx + 1] if idx < len(timesteps) - 1 else 0
        s0, s1 = timesteps[idx], timesteps[idx - 1] if idx > 0 else None
        
        c0 = sigma_t[t] / sigma_t[s0]
        c1 = alpha_t[t] * (np.exp(-(lambda_t[t] - lambda_t[s0])) - 1)
        c2 = 0.5 * c1 * ((lambda_t[t] - lambda_t[s0]) / (lambda_t[s0] - lambda_t[s1])) if idx > 0 else np.array(0.0, dtype="float32")
        
        alphas.append(alpha_t[timestep].item())
        sigmas.append(sigma_t[timestep].item())
        c0s.append(c0.item())
        c1s.append(c1.item())
        c2s.append(c2.item())
        
    const_dict = {}
    const_dict["num_steps"] = len(timesteps)
    const_dict["timesteps"] = timesteps.tolist()
    const_dict["alpha"] = alphas
    const_dict["sigma"] = sigmas
    const_dict["c0"] = c0s
    const_dict["c1"] = c1s
    const_dict["c2"] = c2s
        
    return const_dict
        

def dpm_solver_multistep_convert_model_output(sample, model_output, alpha, sigma):
    return (sample - sigma * model_output) / alpha


def dpm_solver_multistep_scheduler_step(sample, model_output, last_model_output, c0, c1, c2):
    return c0 * sample - c1 * model_output - c2 * (model_output - last_model_output)


def generate_dpm_solver_multistep_scheduler_steps() -> tvm.IRModule:
    def emit_convert_model_output(bb):
        sample = relax.Var("sample", R.Tensor((1, 4, 64, 64), "float32"))
        model_output = relax.Var("model_output", R.Tensor((1, 4, 64, 64), "float32"))
        alpha = relax.Var("alpha", R.Tensor((), "float32")) 
        sigma = relax.Var("sigma", R.Tensor((), "float32"))
        
        with bb.function("dpm_solver_multistep_scheduler_conversion_output", [sample, model_output, alpha, sigma]):
            converted_model_output = bb.emit(dpm_solver_multistep_convert_model_output(sample, model_output, alpha, sigma), "converted_model_output")
            bb.emit_func_output(converted_model_output)
            
    def emit_scheduler_step(bb):        
        sample = relax.Var("sample", R.Tensor((1, 4, 64, 64), "float32"))
        model_output = relax.Var("model_output", R.Tensor((1, 4, 64, 64), "float32"))
        last_model_output = relax.Var("last_model_output", R.Tensor((1, 4, 64, 64), "float32"))
        c0 = relax.Var("c0", R.Tensor((), "float32"))
        c1 = relax.Var("c1", R.Tensor((), "float32")) 
        c2 = relax.Var("c2", R.Tensor((), "float32"))
        
        with bb.function("dpm_solver_multistep_scheduler_step", [sample, model_output, last_model_output, c0, c1, c2]):
            prev_sample = bb.emit(dpm_solver_multistep_scheduler_step(sample, model_output, last_model_output, c0, c1, c2), "prev_sample")
            bb.emit_func_output(prev_sample)
            
    bb = relax.BlockBuilder()
    emit_convert_model_output(bb)
    emit_scheduler_step(bb)
    
    return bb.get()
            

class DPMSolverMultistepScheduler(BaseScheduler):
    constants_file_name = "dpm_solver_multistep_scheduler_constants.json"

    @staticmethod
    def scheduler_steps() -> tvm.IRModule:
        return generate_dpm_solver_multistep_scheduler_steps()
    
    @staticmethod
    def list_step_functions() -> List[str]:
        return [
            "dpm_solver_multistep_scheduler_conversion_output",
            "dpm_solver_multistep_scheduler_step",
        ]
        
    @staticmethod
    def calculate_constants() -> Dict[str, List[tvm.nd.NDArray]]:
        return compute_dpm_solver_multistep_scheduler_consts(1000, 20, 0.00085, 0.012)
        

schedulers_build: List[Type[BaseScheduler]] = [DPMSolverMultistepScheduler, PNDMScheduler]