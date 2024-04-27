import torch
from typing import Dict, List, Tuple
import tvm
from tvm import relax
from models.unet_2d_condition import TVMUNet2DConditionModel
from models.autoencoders.autoencoder_kl import AutoencoderKL


def detect_available_torch_device() -> str:
    if tvm.cuda().exist:
        return "cuda"
    else:
        raise ValueError("Cuda is not available.")


def load_unet_state_dict(pipe) -> Dict[str, torch.Tensor]:
    pt_model_dict = pipe.unet.state_dict()
    model_dict = {}
    for name, tensor in pt_model_dict.items():
        if name.endswith("ff.net.0.proj.weight") or name.endswith("ff.net.0.proj.bias"):
            w1, w2 = tensor.chunk(2, dim=0)
            model_dict[name.replace("proj", "proj1")] = w1
            model_dict[name.replace("proj", "proj2")] = w2
        elif (name.endswith("proj_in.weight") or name.endswith("proj_out.weight")) and len(tensor.shape) == 2:
            model_dict[name] = torch.unsqueeze(torch.unsqueeze(tensor, -1), -1)
        else:
            model_dict[name] = tensor
    return model_dict


def get_unet(pipe, device_str: str, cross_attention_dim=768, attention_head_dim=8, use_linear_projection=False):
    model = TVMUNet2DConditionModel(
        sample_size=64,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        use_linear_projection=use_linear_projection,
        device=device_str,
    )
    model_dict = load_unet_state_dict(pipe)
    model.load_state_dict(model_dict)
    return model


def merge_irmodules(*irmodules: tvm.IRModule) -> tvm.IRModule:
    merged_mod = tvm.IRModule()
    for mod in irmodules:
        for gv, func in mod.functions.items():
            merged_mod[gv] = func
    return merged_mod


def split_irmodules(mod: tvm.IRModule, model_names: List[str], mod_inference_entry_func: List[str]) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_inference = tvm.IRModule()
    transform_func_names = [name + "_transform_params" for name in model_names]

    def assign_func(gv, func):
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_inference[gv] = func
        elif gv.name_hint in transform_func_names:
            mod_transform[gv] = func
        else:
            mod_inference[gv] = func

    for gv in mod.functions:
        func = mod[gv]
        assign_func(gv, func)

    mod_transform = relax.transform.DeadCodeElimination(transform_func_names)(mod_transform)
    mod_inference = relax.transform.DeadCodeElimination(mod_inference_entry_func)(mod_inference)

    return mod_transform, mod_inference


def transform_params(mod_transform: tvm.IRModule, model_params: Dict[str, List[tvm.nd.NDArray]]) -> Dict[str, List[tvm.nd.NDArray]]:
    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())

    def transform_model_params(name, params):
        return vm[name + "_transform_params"](params)

    new_params = {name: transform_model_params(name, params) for name, params in model_params.items()}
    return new_params


def save_params(params: Dict[str, List[tvm.nd.NDArray]], artifact_path: str) -> None:
    from tvm.contrib import tvmjs

    def create_meta_data():
        meta_data = {}
        for model in ["unet", "vae", "clip"]:
            meta_data[f"{model}ParamSize"] = len(params[model])
        return meta_data

    def create_param_dict():
        param_dict = {}
        for model in ["unet", "vae", "clip"]:
            for i, nd in enumerate(params[model]):
                param_dict[f"{model}_{i}"] = nd
        return param_dict

    meta_data = create_meta_data()
    param_dict = create_param_dict()
    tvmjs.dump_ndarray_cache(param_dict, f"{artifact_path}/params", meta_data=meta_data)


def load_params(artifact_path: str, device) -> Dict[str, List[tvm.nd.NDArray]]:
    from tvm.contrib import tvmjs

    def extract_params(model):
        plist = []
        size = meta[f"{model}ParamSize"]
        for i in range(size):
            plist.append(params[f"{model}_{i}"])
        return plist

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    pdict = {model: extract_params(model) for model in ["vae", "unet", "clip"]}
    return pdict


def get_vae(pipe, type):
    def create_model(act_fn, block_out_channels, down_block_types, in_channels, latent_channels, layers_per_block, norm_num_groups, out_channels, sample_size, up_block_types, scaling_factor=None):
        kwargs = {
            "act_fn": act_fn,
            "block_out_channels": block_out_channels,
            "down_block_types": down_block_types,
            "in_channels": in_channels,
            "latent_channels": latent_channels,
            "layers_per_block": layers_per_block,
            "norm_num_groups": norm_num_groups,
            "out_channels": out_channels,
            "sample_size": sample_size,
            "up_block_types": up_block_types
        }
        if scaling_factor is not None:
            kwargs["scaling_factor"] = scaling_factor
        return AutoencoderKL(**kwargs)

    if type == "1.5":
        model = create_model(
            act_fn="silu",
            block_out_channels=[128, 256, 512, 512],
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
        )
    elif type == "XL":
        model = create_model(
            act_fn="silu",
            block_out_channels=[128, 256, 512, 512],
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=1024,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            scaling_factor=0.13025
        )
    else:
        raise ValueError(f"Unsupported VAE type: {type}")

    model.load_state_dict(pipe.vae.state_dict())
    return model