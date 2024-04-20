import torch
import torch.fx as fx
import tvm
from tvm import relax, te
from tvm.relax.frontend.torch import dynamo_capture_subgraphs, from_fx
from tvm.script import relax as R

import utils.utils as utils


def build_clip_container(clip_module):
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, text_input_ids):
            return self.clip(text_input_ids)[0]

    return TextEncoderWrapper(clip_module)


def trace_clip_operations(clip_wrapper):
    input_ids = torch.rand((1, 77)).to(torch.int32)
    captured_mod = dynamo_capture_subgraphs(
        clip_wrapper.forward, input_ids, keep_params_as_input=True
    )
    assert len(captured_mod.functions) == 1
    return tvm.IRModule({"clip": captured_mod["subgraph_0"]})


def convert_clip_to_embeddings(pipe):
    clip_module = pipe.text_encoder
    clip_wrapper = build_clip_container(clip_module)
    return trace_clip_operations(clip_wrapper)


def build_unet_container(unet_module, scale):
    class UnetWrapper(torch.nn.Module):
        def __init__(self, unet, scale):
            super().__init__()
            self.unet = unet
            self.scale = scale

        def forward(self, latents, timestep, text_embeddings):
            latent_input = torch.cat([latents] * 2, dim=0)
            noise_pred = self.unet(latent_input, timestep, text_embeddings)
            uncond_pred, text_pred = noise_pred.chunk(2)
            return uncond_pred + self.scale * (text_pred - uncond_pred)

    return UnetWrapper(unet_module, scale)


def trace_unet_operations(unet_wrapper, attn_dim):
    unet_graph = fx.symbolic_trace(unet_wrapper)
    return from_fx(
        unet_graph,
        [
            ((1, 4, 64, 64), "float32"),
            ((), "int32"),
            ((2, 77, attn_dim), "float32"),
        ],
        keep_params_as_input=True,
    )


def generate_noise_predictions(pipe, device_str: str):
    unet_config = pipe.unet.config
    attn_dim = unet_config.cross_attention_dim
    head_dim = unet_config.attention_head_dim
    use_linear_proj = unet_config.get("use_linear_projection")

    unet_module = utils.get_unet(
        pipe,
        device_str,
        cross_attention_dim=attn_dim,
        attention_head_dim=head_dim,
        use_linear_projection=use_linear_proj,
    )

    scale = 7.5
    unet_wrapper = build_unet_container(unet_module, scale)
    traced_mod = trace_unet_operations(unet_wrapper, attn_dim)
    return tvm.IRModule({"unet": traced_mod["main"]})


def create_vae_wrapper(vae_module):
    class VaeWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            scaled_latents = 1 / 0.18215 * latents
            decoded_image = self.vae.decode(scaled_latents, return_dict=False)[0]
            normalized_image = (decoded_image / 2 + 0.5).clamp(min=0, max=1)
            reshaped_image = (normalized_image.permute(0, 2, 3, 1) * 255).round()
            return reshaped_image

    return VaeWrapper(vae_module)


def perform_vae_tracing(vae_wrapper):
    vae_graph = fx.symbolic_trace(vae_wrapper)
    return from_fx(vae_graph, [((1, 4, 64, 64), "float32")], keep_params_as_input=True)


def vae_to_image(pipe):
    vae_module = utils.get_vae(pipe, "1.5")
    vae_wrapper = create_vae_wrapper(vae_module)
    traced_mod = perform_vae_tracing(vae_wrapper)
    return tvm.IRModule({"vae": traced_mod["main"]})


def define_image_to_rgba_compute(input_tensor):
    def _compute_func(y, x):
        return (
            input_tensor[0, y, x, 0].astype("uint32")
            | (input_tensor[0, y, x, 1].astype("uint32") << 8)
            | (input_tensor[0, y, x, 2].astype("uint32") << 16)
            | tvm.tir.const(255 << 24, "uint32")
        )

    return te.compute((512, 512), _compute_func, name="image_to_rgba")


def build_image_to_rgba_module(compute_func):
    builder = relax.BlockBuilder()
    input_var = relax.Var("input", R.Tensor([1, 512, 512, 3], "float32"))

    with builder.function("image_to_rgba", [input_var]):
        with builder.dataflow():
            output_var = builder.emit_output(
                builder.call_te(
                    compute_func, input_var, primfunc_name_hint="tir_image_to_rgba"
                )
            )
        builder.emit_func_output(output_var)

    return builder.get()


def image_to_rgba():
    compute_func = define_image_to_rgba_compute
    return build_image_to_rgba_module(compute_func)


def build_combine_embeddings_module():
    builder = relax.BlockBuilder()
    cond_var = relax.Var("cond", R.Tensor([1, 77, 768], "float32"))
    uncond_var = relax.Var("uncond", R.Tensor([1, 77, 768], "float32"))

    with builder.function("combine_embeddings", [cond_var, uncond_var]):
        with builder.dataflow():
            output_var = builder.emit_output(
                relax.op.concat([cond_var, uncond_var], axis=0)
            )
        builder.emit_func_output(output_var)

    return builder.get()


def combine_embeddings():
    return build_combine_embeddings_module()