"""Model loading, linear probes, activation capture and steering hooks for the GPU stages."""
import os
from omegaconf import DictConfig
from typing import Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor
import torch


class LinearProbe(torch.nn.Module):
    """Linear probe on one activation vector: maps inputs of shape [n, in_dim] to logits [n, 1].

    The weight `w` ([in_dim, 1]) is the trait direction used by the analyses and for steering.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(in_dim, 1, dtype=torch.float32) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b

def load_probes(probe_dir, probe_checkpoint, trait_id, set_name, submodule, pooling, l1, layers):
    """Load the trained probes of one trait for the given layers.

    Returns {layer: LinearProbe} with every probe moved to the GPU. Layers without a probe file are
    left out of the mapping (no error).
    """
    probes = {}
    for layer in layers:
        path = os.path.join(probe_dir, probe_checkpoint, f"trait_{trait_id}", set_name, "log_probes", submodule, pooling, f"l1_{l1}", f"layer_{layer}.pt")
        if os.path.exists(path):
            state = torch.load(path, weights_only=True)
            in_dim = state['w'].shape[0]
            probe = LinearProbe(in_dim)
            probe.load_state_dict(state)
            probes[layer] = probe.cuda()
    return probes
    


def load_model_and_tokenizer(cfg: DictConfig) -> Tuple[Any, Any]:
    """Return (model, tokenizer) for `cfg.model.checkpoint`, as defined in configs/checkpoints.yaml.

    The checkpoint is loaded from its Hugging Face repo (`hf_repo`, optional `hf_branch`) or a local
    `path`, with `cfg.model.dtype` and `cfg.model.device_map`. apertus_v1.5_8B is loaded through
    AutoProcessor/AutoModelForMultimodalLM, all other checkpoints as causal LMs. The tokenizer pads on
    the left and uses the EOS token as pad token if none is set.
    """
    checkpoint_cfg = cfg.checkpoints[cfg.model.checkpoint]

    if checkpoint_cfg.get("hf_repo"):
        checkpoint = checkpoint_cfg.hf_repo
        revision = checkpoint_cfg.get("hf_branch")
        extra = {"revision": revision} if revision is not None else {}

    elif checkpoint_cfg.get("path"):
        checkpoint = checkpoint_cfg.path
        extra = {}

    else:
        raise ValueError(f"Checkpoint '{cfg.model.checkpoint}' must define either 'hf_repo' or 'path'.")

    dtype = getattr(torch, cfg.model.dtype)

    if cfg.model.checkpoint == "apertus_v1.5_8B":
        processor = AutoProcessor.from_pretrained(checkpoint, **extra)
        tokenizer = processor.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        model = AutoModelForMultimodalLM.from_pretrained(
            checkpoint,
            device_map=cfg.model.device_map,
            dtype=dtype,
            **extra,
        )
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **extra)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map=cfg.model.device_map,
        dtype=dtype,
        **extra,
    )
    return model, tokenizer



def get_num_hidden_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model.config, "text_config"):
        return model.config.text_config.num_hidden_layers
    raise AttributeError(f"Cannot determine number of hidden layers for {type(model.config).__name__}")



def get_transformer_layers(model):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"Cannot locate transformer layers for {type(model).__name__}")


def collect_hooks(layer, layer_id: int, submodules: list, store: dict):
    """Register hooks for real submodules and virtual residuals in one interface."""
    handles = []
    
    for sub in submodules:
        if sub == "residual.pre_attn":
            def pre_hook(module, inp, subname=sub):
                store[(layer_id, subname)] = inp[0].detach().cpu()
            handles.append(layer.register_forward_pre_hook(pre_hook))

        elif sub == "residual.post_attn":
            def post_attn_hook(module, inp, out, subname=sub):
                store[(layer_id, subname)] = out.detach().cpu()
            handles.append(layer.attention_layernorm.register_forward_hook(post_attn_hook))

        elif sub == "residual.post_mlp":
            def post_mlp_hook(module, inp, out, subname=sub):
                store[(layer_id, subname)] = out.detach().cpu()
            handles.append(layer.register_forward_hook(post_mlp_hook))

        else:
            parts = sub.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)
            def sub_hook(module, inp, out, subname=sub):
                store[(layer_id, subname)] = out.detach().cpu()
            handles.append(target.register_forward_hook(sub_hook))

    return handles


def register_steering_hook(layer, layer_id: int, submodule: str, direction: torch.Tensor, strength: float):
    """Norm-relative additive steering: h' = h + strength * ||h|| * unit(direction).

    Mirrors collect_hooks's submodule branching, but returns a modified tensor
    instead of storing it, so the hook actually intervenes on the forward pass.
    """
    unit = direction / direction.norm()

    def steer(h: torch.Tensor) -> torch.Tensor:
        return h + strength * h.norm(dim=-1, keepdim=True) * unit.to(h.dtype).to(h.device)

    handles = []

    if submodule == "residual.pre_attn":
        def pre_hook(module, inp):
            return (steer(inp[0]),) + inp[1:]
        handles.append(layer.register_forward_pre_hook(pre_hook))

    elif submodule == "residual.post_attn":
        def post_attn_hook(module, inp, out):
            return steer(out)
        handles.append(layer.attention_layernorm.register_forward_hook(post_attn_hook))

    elif submodule == "residual.post_mlp":
        def post_mlp_hook(module, inp, out):
            return steer(out)
        handles.append(layer.register_forward_hook(post_mlp_hook))

    else:
        parts = submodule.split(".")
        target = layer
        for p in parts:
            target = getattr(target, p)
        def sub_hook(module, inp, out):
            return steer(out)
        handles.append(target.register_forward_hook(sub_hook))

    return handles


def activation_pooling(h: torch.Tensor, span: tuple, mode: str) -> torch.Tensor:
    """Pool the activations `h` ([seq, d]) of one sequence over its generated tokens; returns [d].

    Modes: "last_pooling" (last generated token), "mean_pooling" (mean) and "max_pooling"
    (element-wise max). Any other mode returns None. If nothing was generated, zeros are returned.

    `span` is the half-open (start, end) interval of generated-token positions in `h`, as
    returned by generate_and_capture_activations. Because the tokenizer pads on the left,
    `start` is the *padded* prompt length -- identical for every row in the batch -- not the
    row's own unpadded token count; `end` excludes the padding `generate` appends after a
    sequence finishes early. Indexing with the unpadded length instead would fold the padding
    and the trait instruction itself into the mean, leaking the label into the activation.
    """
    start, end = span
    if end <= start:  # nothing was generated
        return torch.zeros_like(h[0])

    if mode == "last_pooling":
        return h[end - 1]  # [d], last generated token
    if mode == "mean_pooling":
        return h[start:end].mean(dim=0)
    if mode == "max_pooling":
        return h[start:end].max(dim=0).values


def generate_and_capture_activations(model, tokenizer, prompts, layers, submodules, max_new_tokens, temperature):
    """Generate completions for a batch of chat prompts, then record activations on the full sequences.

    Returns (gen_ids, gen_spans, store):
      gen_ids    [batch, seq] token ids of prompt + completion (left-padded prompts);
      gen_spans  per row, the half-open (start, end) positions of the generated tokens (see
                 activation_pooling);
      store      {(layer, submodule): tensor [batch, seq, d] on the CPU} from a second forward pass.
    Sampling is used if `temperature` > 0, greedy decoding otherwise.
    """
    texts = []
    if tokenizer.chat_template is not None:
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    else:
        # Fallback: manually format the chat messages
        for p in prompts:
            text = ""
            for msg in p:
                if msg["role"] == "system":
                    text += f"{msg['content']}\n\n"
                elif msg["role"] == "user":
                    text += f"User: {msg['content']}\n\nAssistant:"
            text += " "
            texts.append(text)
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    # Padding is on the left, so generation starts at the padded width for every row -- this,
    # not attention_mask.sum(1), is the prompt/completion boundary in the refed sequence.
    gen_start = inputs["input_ids"].shape[1]

    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=temperature if temperature > 0 else None)

    # Per-row count of real generated tokens: everything before the first pad token that
    # `generate` appends once a sequence has finished.
    gen_part = gen_ids[:, gen_start:]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        n_gen = torch.full((gen_ids.shape[0],), gen_part.shape[1], device=gen_ids.device, dtype=torch.long)
    else:
        is_pad = gen_part == pad_id
        first_pad = is_pad.int().argmax(dim=1)
        n_gen = torch.where(is_pad.any(dim=1), first_pad, torch.full_like(first_pad, gen_part.shape[1]))

    gen_spans = [(gen_start, gen_start + int(n)) for n in n_gen.cpu().tolist()]

    # Refeed the full prompt + completions for activations. Pass an explicit attention mask so
    # the left padding and the trailing generation padding are not attended to as real tokens.
    positions = torch.arange(gen_part.shape[1], device=gen_ids.device)
    gen_attention = (positions.unsqueeze(0) < n_gen.unsqueeze(1)).long()
    full_attention = torch.cat([inputs["attention_mask"], gen_attention], dim=1)

    with torch.no_grad():
        store = {}
        handles = []
        transformer_layers = get_transformer_layers(model)
        for l in layers:
            handles += collect_hooks(transformer_layers[l], l, submodules, store)
        _ = model(gen_ids, attention_mask=full_attention, return_dict=True)
        for h in handles:
            h.remove()

    return gen_ids, gen_spans, store


