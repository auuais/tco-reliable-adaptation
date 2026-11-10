"""Keep immutable FP32 weights on the host and reuse their ordinary autocast operands."""
from types import MethodType

import torch
import torch.nn.functional as F


def _forward(layer,x):
    if torch.is_grad_enabled() and (layer.weight.requires_grad or (layer.bias is not None and layer.bias.requires_grad)):
        raise RuntimeError("This storage mode requires frozen base weights")
    if layer.weight._version!=layer._tco_versions[0] or (layer.bias is not None and layer.bias._version!=layer._tco_versions[1]):
        raise RuntimeError("Frozen parameters changed after constructing autocast operands")
    if (torch.is_autocast_enabled(x.device.type) and torch.get_autocast_dtype(x.device.type)==layer._tco_weight.dtype
            and x.dtype in (torch.float32,layer._tco_weight.dtype) and x.device==layer._tco_weight.device):
        return F.linear(x,layer._tco_weight,layer._tco_bias)
    return F.linear(x,layer.weight.to(x.device),None if layer.bias is None else layer.bias.to(x.device))


def cache_linear(linear,operand_dtype):
    if (type(linear) is not torch.nn.Linear or hasattr(linear,"_tco_weight") or linear.weight.dtype!=torch.float32
            or operand_dtype not in (torch.float16,torch.bfloat16) or any(p.requires_grad for p in linear.parameters())):
        raise ValueError("Expected an unmodified frozen FP32 linear projection")
    dest=linear.weight.device
    raw_bytes=sum(p.numel()*p.element_size() for p in linear.parameters())
    linear.to("cpu")
    shadow=linear.weight.detach().to(device=dest,dtype=operand_dtype)
    bias=None if linear.bias is None else linear.bias.detach().to(device=dest,dtype=operand_dtype)
    linear.register_buffer("_tco_weight",shadow,persistent=False)
    linear.register_buffer("_tco_bias",bias,persistent=False)
    linear._tco_versions=(linear.weight._version,None if linear.bias is None else linear.bias._version)
    linear.forward=MethodType(_forward,linear)
    return raw_bytes,shadow.numel()*shadow.element_size()+(0 if bias is None else bias.numel()*bias.element_size())


def enable(model,operand_dtype):
    from peft.lora import LoRALayer
    selected=[m for blocks in (model.aggregator.frame_blocks,model.aggregator.global_blocks)
              for m in blocks.modules() if isinstance(m,LoRALayer)]
    if not selected:raise ValueError("Call after upstream adapter initialization")
    adapter_types={str(p.dtype) for m in selected for p in (m.lora_A,m.lora_B)}
    if adapter_types!={"torch.float32"}:raise ValueError("Unexpected adapter precision")
    sizes=[cache_linear(m.original_layer,operand_dtype) for m in selected]
    if any(m.original_layer.weight.device.type!="cpu" for m in selected):raise RuntimeError("CPU master missing")
    torch.clear_autocast_cache()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(projections=len(selected),fp32_master_bytes=sum(a for a,_ in sizes),
                cached_operand_bytes=sum(b for _,b in sizes),adapter_dtype="float32",operand_dtype=str(operand_dtype))
