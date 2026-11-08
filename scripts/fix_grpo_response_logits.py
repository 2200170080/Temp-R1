#!/usr/bin/env python3
"""Reduce Qwen GRPO logits memory after checking CPU output/gradient equivalence.

Run using the existing tempr1 environment. Does not launch/stop training,
change optimizer settings, or modify SFT weights.
"""
import argparse
from datetime import datetime
import inspect
from pathlib import Path
import re
import shutil
import sys


def check_equivalence():
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM

    if "logits_to_keep" not in inspect.signature(Qwen2ForCausalLM.forward).parameters:
        raise RuntimeError("Installed Qwen forward does not support logits_to_keep; no files changed.")
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).cpu().float().eval()
    inputs = torch.randint(3, 32, (2, 12), device="cpu")
    response_length = 5
    labels = inputs[:, -response_length:]
    weights = torch.randn(2, response_length, device="cpu")

    def run(limited):
        model.zero_grad(set_to_none=True)
        kwargs = {"logits_to_keep": response_length + 1} if limited else {}
        result = model(input_ids=inputs, use_cache=False, **kwargs)
        logits = result.logits[:, -response_length - 1:-1, :]
        log_probs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        entropy = torch.logsumexp(logits, dim=-1) - (logits.softmax(dim=-1) * logits).sum(dim=-1)
        loss = -(log_probs * weights).mean() - 0.001 * entropy.mean()
        loss.backward()
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        return logits.detach().clone(), loss.detach().clone(), grads

    original, optimized = run(False), run(True)
    torch.testing.assert_close(original[0], optimized[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(original[1], optimized[1], rtol=1e-5, atol=1e-6)
    if original[2].keys() != optimized[2].keys():
        raise RuntimeError("Parameter gradient sets differ; no files changed.")
    for name in original[2]:
        torch.testing.assert_close(original[2][name], optimized[2][name], rtol=3e-5, atol=3e-6,
                                   msg=lambda message: name + ": " + message)
    print("CPU_EQUIVALENCE_OK: response logits, loss, and all parameter gradients match.")
    print("Transformers:", transformers.__version__, "Torch:", torch.__version__)


def patched_source(text):
    marker = "# GRPO response-only logits: preserve full context, skip unused prompt projection."
    if marker in text:
        return None
    pattern = re.compile(
        r"(?P<indent>^[ \t]*)output = self\.actor_module\(input_ids=input_ids,\n"
        r"[ \t]*attention_mask=attention_mask,\n"
        r"[ \t]*position_ids=position_ids,\n"
        r"[ \t]*use_cache=False\)(?P<comment>[^\n]*)",
        re.MULTILINE,
    )
    if len(list(pattern.finditer(text))) != 1:
        raise RuntimeError("Actor source differs from expected non-padding forward block; no files changed.")

    def replace(match):
        pad = match.group("indent")
        # Restrict the new argument to Qwen2 models exposing the supported API.
        lines = [
            marker,
            'forward_kwargs = {}',
            'if getattr(self.actor_module.config, "model_type", None) == "qwen2":',
            '    forward_kwargs["logits_to_keep"] = response_length + 1',
            'output = self.actor_module(',
            '    input_ids=input_ids,',
            '    attention_mask=attention_mask,',
            '    position_ids=position_ids,',
            '    use_cache=False,',
            '    **forward_kwargs,',
            ')',
        ]
        return "\n".join(pad + line for line in lines)

    updated = pattern.sub(replace, text, count=1)
    compile(updated, "dp_actor.py", "exec")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "Temp-R1")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    path = root / "verl/workers/actor/dp_actor.py"
    if path.is_symlink() or root not in path.resolve().parents:
        raise RuntimeError("Actor source is outside the expected project location.")
    text = path.read_text(encoding="utf-8")
    updated = patched_source(text)
    if updated is None:
        print("Patch already installed; no changes needed.")
        return
    check_equivalence()
    backup = root.parent / ("Temp-R1-grpo-memory-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup.mkdir()
    shutil.copy2(str(path), str(backup / "dp_actor.py"))
    temporary = path.with_name(".dp_actor.memory-patch.tmp")
    temporary.write_text(updated, encoding="utf-8")
    shutil.copymode(str(path), str(temporary))
    # Avoid replacing edits made after the initial read.
    if path.read_text(encoding="utf-8") != text:
        temporary.unlink()
        raise RuntimeError("Actor source changed during validation; stopped without patching.")
    temporary.replace(path)
    print("Original actor source backed up to:", backup / "dp_actor.py")
    print("Patched:", path)
    print("GRPO_MEMORY_PATCH_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("STOPPED:", error, file=sys.stderr)
        raise
