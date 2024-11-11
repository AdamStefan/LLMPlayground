import torch
from networks.gpt.gpt_model import GPTModel, Config


def load_from_hf_weights(model_type: str):

    config_args_dict = {
        "gpt2": dict(n_layers=12, n_heads=12, dim=768),  # 124M params
        "gpt2-medium": dict(n_layers=24, n_heads=16, dim=1024),  # 350M params
        "gpt2-large": dict(n_layers=36, n_heads=20, dim=1280),  # 774M params
        "gpt2-xl": dict(n_layers=48, n_heads=25, dim=1600),  # 1558M params
    }[model_type]

    from transformers import GPT2LMHeadModel

    config_args_dict["vocab_size"] = 50257
    config_args_dict["block_size"] = 1024
    config_args_dict["bias"] = True
    print("Creating GPT model")
    config = Config(**config_args_dict)
    gpt = GPTModel(config=config)
    sd = gpt.state_dict()
    print("Loading hugging face model")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    print("Copying hugging face weighs")

    sd_key_mappings = {
        "embeddings.weight": "transformer.wte.weight",
        "absolute_positional_embeddings.weight": "transformer.wpe.weight",
        "ln_after_transformer.weight": "transformer.ln_f.weight",
        "ln_after_transformer.bias": "transformer.ln_f.bias",
        "head_projection.weight": "lm_head.weight",
    }

    # map all transformer blocks
    for i in range(config.n_layers):
        sd_key_mappings[f"transformer_blocks.{i}.ln_1.weight"] = f"transformer.h.{i}.ln_1.weight"
        sd_key_mappings[f"transformer_blocks.{i}.ln_1.bias"] = f"transformer.h.{i}.ln_1.bias"
        sd_key_mappings[f"transformer_blocks.{i}.attn.QKV.weight"] = f"transformer.h.{i}.attn.c_attn.weight"
        sd_key_mappings[f"transformer_blocks.{i}.attn.QKV.bias"] = f"transformer.h.{i}.attn.c_attn.bias"
        sd_key_mappings[f"transformer_blocks.{i}.attn.proj.weight"] = f"transformer.h.{i}.attn.c_proj.weight"
        sd_key_mappings[f"transformer_blocks.{i}.attn.proj.bias"] = f"transformer.h.{i}.attn.c_proj.bias"
        sd_key_mappings[f"transformer_blocks.{i}.ln_2.weight"] = f"transformer.h.{i}.ln_2.weight"
        sd_key_mappings[f"transformer_blocks.{i}.ln_2.bias"] = f"transformer.h.{i}.ln_2.bias"
        sd_key_mappings[f"transformer_blocks.{i}.mlp.c_fc.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
        sd_key_mappings[f"transformer_blocks.{i}.mlp.c_fc.bias"] = f"transformer.h.{i}.mlp.c_fc.bias"
        sd_key_mappings[f"transformer_blocks.{i}.mlp.c_proj.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"
        sd_key_mappings[f"transformer_blocks.{i}.mlp.c_proj.bias"] = f"transformer.h.{i}.mlp.c_proj.bias"

    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

    for sd_key, hf_key in sd_key_mappings.items():
        if any(hf_key.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[hf_key].shape[::-1] == sd[sd_key].shape
            with torch.no_grad():
                sd[sd_key].copy_(sd_hf[hf_key].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[hf_key].shape == sd[sd_key].shape
            with torch.no_grad():
                sd[sd_key].copy_(sd_hf[hf_key])

    print("Weights loaded.")

    return gpt


def test_model_generate(input_str, gpt: GPTModel, device: torch.device):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    gpt.eval()
    input_ids = tokenizer.encode(input_str, allowed_special={"<|endoftext|>"})
    input_ids = torch.Tensor(input_ids).to(device).unsqueeze(0).long()
    out = gpt.generate(input_ids, max_new_tokens=100)
    out_token_ids = out.detach().cpu().squeeze(0).tolist()
    out_text = tokenizer.decode(out_token_ids)
    print(out_text)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpt = load_from_hf_weights("gpt2")
    gpt.to(device=device)
    test_model_generate("How are you? I am", gpt, device=device)
