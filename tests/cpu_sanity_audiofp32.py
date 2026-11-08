"""CPU sanity test for the Star7 FP32-carrier patch on a tiny native H3 model."""
import importlib.util, sys, types
sys.argv = ["test", "--cpu"]
sys.path.insert(0, "/workspace/comfy_wf/repos/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing()
import torch
import comfy.ops, comfy.patcher_extension
import comfy.ldm.minimax.model as mm

spec = importlib.util.spec_from_file_location("star7_patched", "/workspace/kaggle/star7_audio_patch/nodes.py")
star7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(star7)
print("FP32_CARRIER =", star7.FP32_CARRIER)

torch.manual_seed(0)
H = 128
model = mm.MiniMaxH3Model(hidden_size=H, num_layers=2, token_refiner_num_layers=1, num_attention_heads=1,
                          attention_head_dim=128, ffn_hidden_size=128, text_dim=H, timestep_input_dim=32,
                          time_embed_hidden_size=64, time_embed_dim=64, dtype=torch.float16,
                          operations=comfy.ops.manual_cast)
for n, p in model.named_parameters():
    with torch.inference_mode():
        p.copy_(torch.randn_like(p, dtype=torch.float32).mul(0.02).to(p.dtype))
with torch.no_grad():
    model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16, dtype=torch.float32) / 16)))

model.requires_grad_(False)
video = torch.randn(1, 24, 2, 4, 4, dtype=torch.float16)
audio = torch.randn(1, 32, 2, 5, dtype=torch.float16)
ctx = torch.randn(1, 6, H, dtype=torch.float16)
ts = torch.tensor([700.0])

seen = {}
def spy_wrapper(executor, x, timestep, context, transformer_options={}, *a, **k):
    # stands in for Spectrum: records what dtype it would forecast in
    seen["wrapper_ctx"] = context.dtype
    seen["wrapper_x"] = tuple(t.dtype for t in x)
    return executor(x, timestep, context, transformer_options, *a, **k)

orig_core = model._forward
def core_spy(x, timestep, context, transformer_options={}, *a, **k):
    seen["core_ctx"] = context.dtype
    return orig_core(x, timestep, context, transformer_options, *a, **k)

def run(patch):
    opts = {"wrappers": {comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {}}}
    w = opts["wrappers"][comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    if patch:
        w["star7"] = [star7._commercial_te_boundary_wrapper]
        model._forward = types.MethodType(star7._inner_forward(core_spy), model)
        model.final_layer.forward = types.MethodType(
            star7._final_layer_forward(mm.FinalLayer.forward.__get__(model.final_layer)), model.final_layer)
    else:
        model._forward = core_spy
        model.final_layer.forward = mm.FinalLayer.forward.__get__(model.final_layer)
    w["spectrum_spy"] = [spy_wrapper]
    with torch.inference_mode():
        return model.forward([video.clone(), audio.clone()], ts, ctx, transformer_options=opts,
                             minimax_payload={"audio_scale": 4.0})

base = run(False); b = dict(seen)
pat = run(True); p = dict(seen)
print("baseline :", b)
print("patched  :", p)
for name, x0, x1 in (("video", base[0], pat[0]), ("audio", base[1], pat[1])):
    d = (x0.float() - x1.float()).abs().max().item()
    print(f"{name}: base dtype={x0.dtype} patched dtype={x1.dtype} max|diff|={d:.3e} max|v|={x1.float().abs().max().item():.3f}")
assert p["wrapper_ctx"] == torch.float32 and p["core_ctx"] == torch.float16
assert b["wrapper_ctx"] == torch.float16
assert torch.isfinite(pat[0]).all() and torch.isfinite(pat[1]).all()

# Spectrum-style overflow demo: forecast tensor written into context dtype
resid = torch.tensor([1.0, -3.0, 120000.0, 0.5])
print("fp16 forecast of residual row:", resid.to(torch.float16).tolist(), "-> clamp ->",
      torch.nan_to_num(resid.to(torch.float16), posinf=65504., neginf=-65504.).tolist())
print("ALL OK")
