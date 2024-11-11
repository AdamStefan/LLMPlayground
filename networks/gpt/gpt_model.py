import torch
import math
from torch import Tensor, nn
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Config:
    dim: int
    n_heads: int = 12
    dropout: float = 0
    bias: bool = False
    use_flash: bool = True
    n_layers: int = 12
    block_size: int = 1024
    vocab_size: int = 50257


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input) -> Tensor:
        return nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiheadSelfAttention, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.QKV = nn.Linear(self.dim, self.dim * 3, bias=config.bias)
        self.dropout = config.dropout
        self.attention_dropout = nn.Dropout(self.dropout)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.proj = nn.Linear(self.dim, self.dim)
        self.use_flash = config.use_flash
        if not self.use_flash:
            b_size = config.block_size
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(b_size, b_size)).view(1, 1, b_size, b_size))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()

        q, k, v = self.QKV(x).split(self.dim, dim=2)

        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        if self.use_flash:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attention = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            attention = nn.functional.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)
            out = attention @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.residual_dropout(out)

        return out


class MLP(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim, bias=config.bias)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.dim, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> Tensor:
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim, bias=config.bias)
        self.attn = MultiheadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x) -> Tensor:
        out = x + self.attn(self.ln_1(x))
        out = out + self.mlp(self.ln_2(out))
        return out


class GPTModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.absolute_positional_embeddings = nn.Embedding(config.block_size, config.dim)
        self.transformer_blocks = nn.ModuleList()
        self.ln_after_transformer = nn.LayerNorm(config.dim, config.bias)
        for i in range(config.n_layers):
            block = Block(config=config)
            self.transformer_blocks.append(block)
        self.head_projection = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head_projection.SKIP_INIT = 1  # don't init this one, we will tie weights

    def forward(self, input_ids: Tensor, targets=None, return_logits=True) -> Tuple[Tensor | None, Tensor | None]:

        B, T = input_ids.size()
        x = self.embeddings(input_ids)
        positions = torch.arange(0, T, device=x.device)
        pe = self.absolute_positional_embeddings(positions)
        x = x + pe

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln_after_transformer(x)
        logits = self.head_projection(x)

        loss: Tensor | None = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def test_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpt2_mini_config = Config(n_layers=12, n_heads=12, dim=768)
    gpt = GPTModel(config=gpt2_mini_config)
    gpt = gpt.to(device=device)
    input_ids = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]).to(device=device).long()
    targets = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]).to(device=device).long()
    out = gpt(input_ids, targets=targets)


if __name__ == "__main__":
    test_model()
