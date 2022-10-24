import math
from functools import partial
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops_exts import repeat_many
from einops.layers.torch import Rearrange, Reduce

from .knn_memory import KNNMemoryList, DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY

# helper functions
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def identity(t):
    return t


def exists(val):
    return val is not None


def unique(arr):
    return list({el: True for el in arr}.keys())


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def l2norm(t):
    return F.normalize(t, dim=-1)


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True).detach()
    return F.softmax(t, dim=dim)


# helper classes


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        out = self.fn(self.norm(x), **kwargs)

        if not isinstance(out, tuple):
            return out + x

        head, *tail = out
        return (head + x, *tail)


# t5 relative positional bias


class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return bias * self.scale


# feedforward


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        xl_max_memories=0.0,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, *, xl_memory=None, rel_pos_bias=None):
        h, device = self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl_mem, k), dim=-2)
            v = torch.cat((v_xl_mem, v), dim=-2)

        sim = einsum("b h i d, b j d -> b h i j", q, k)
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim
        # modify --> no causal mask
        # causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=device)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # new xl memories

        new_kv_memories = torch.stack((k, v), dim=-2).detach()

        if self.xl_max_memories > 0:
            new_xl_kv_memories = new_kv_memories[:, -self.xl_max_memories :]
        else:
            new_xl_kv_memories = None

        return self.to_out(out), new_xl_kv_memories


# approximate nearest neighbor attention


class KNNAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        num_retrieved_memories=32,
        xl_max_memories=0.0,
        attn_scale_init=20,
    ):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))

        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self, x, *, knn_memory, xl_memory=None, add_knn_memory=True, rel_pos_bias=None
    ):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # in paper, they showed normalizing of keys led to more stable training
        # we'll just go with full cosine sim attention https://arxiv.org/abs/2010.04245

        q, k = map(l2norm, (q, k))

        # handle xl memory

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl_mem, k), dim=-2)
            v = torch.cat((v_xl_mem, v), dim=-2)

        # calculate local attention

        scale = self.scale.exp()

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        mask_value = -torch.finfo(sim.dtype).max

        # causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=device)
        sim = sim.masked_fill(causal_mask, mask_value)

        # calculate knn attention over memory, if index is passed in

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim=-2)

        sim_mem = einsum("b h i d, b h i j d -> b h i j", q, mem_k) * scale
        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)

        # calculate new XL memories, as well as memories to be discarded

        new_kv_memories = torch.stack((k, v), dim=-2).detach()

        if self.xl_max_memories > 0:
            new_kv_memories_discarded, new_xl_kv_memories = (
                new_kv_memories[:, : -self.xl_max_memories],
                new_kv_memories[:, -self.xl_max_memories :],
            )
        else:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(new_kv_memories_discarded)

        # attention (combining local and distant)

        sim = torch.cat((sim_mem, sim), dim=-1)
        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        local_attn, mem_attn = (
            attn[..., self.num_retrieved_memories :],
            attn[..., : self.num_retrieved_memories],
        )
        local_out = einsum("b h i j, b j d -> b h i d", local_attn, v)
        mem_out = einsum("b h i j, b h i j d -> b h i d", mem_attn, mem_v)

        out = local_out + mem_out

        # combine heads and project out

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out), new_xl_kv_memories


# main class


class MemorizingTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # number of tokens should be 256 for us?
        dim,  # token dimensions
        depth,
        num_classes=2,
        dim_head=64,
        heads=16,
        knn_attn_heads=None,
        attn_dropout=0.0,
        ff_mult=4,
        ff_dropout=0.0,
        memorizing_layers=None,
        max_knn_memories=250000,  # max number of tokens in memory
        num_retrieved_memories=32,
        clear_memories_on_sos_token_id=None,
        clear_memories_on_eos_token_id=None,
        knn_memories_directory=DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        shift_knn_memories_down=0.0,
        pad_id=0,
        xl_max_memories=0,
        xl_memory_layers=None,
        shift_xl_memories_down=0.0,
        knn_memory_multiprocessing=False,
        image_size=256,
        patch_size=16,
        channels=3,
        global_pool=False,
    ):
        super().__init__()
        
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pad_id = pad_id

        block_wrapper = partial(PreNormResidual, dim)
        valid_layers = set(range(1, depth + 1))

        memorizing_layers = default(
            memorizing_layers, (depth // 2,)
        )  # default KNN attention layer to midpoint of transformer
        memorizing_layers = cast_tuple(memorizing_layers)
        memorizing_layers = tuple(
            filter(lambda i: i in valid_layers, memorizing_layers)
        )

        self.dim_head = dim_head
        
        self.global_pool = global_pool
        
        knn_attn_heads = default(knn_attn_heads, heads)

        # xl memory hyperparameter

        if xl_max_memories > 0:
            xl_memory_layers = default(xl_memory_layers, tuple(range(1, depth + 1)))
            xl_memory_layers = unique(xl_memory_layers)
            self.xl_memory_layers = tuple(
                filter(lambda i: i in valid_layers, xl_memory_layers)
            )
            self.num_xl_memory_layers = len(self.xl_memory_layers)
        else:
            self.xl_memory_layers = tuple()
            self.num_xl_memory_layers = 0

        # knn memory hyperparameters

        self.max_knn_memories = max_knn_memories
        self.knn_memories_directory = knn_memories_directory
        self.memorizing_layers = unique(memorizing_layers)
        self.num_memory_layers = len(memorizing_layers)

        # print("memorizing_layers", self.memorizing_layers)

        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(scale=dim_head**0.5, heads=heads)
        self.knn_rel_pos_bias = T5RelativePositionBias(
            scale=dim_head**0.5, heads=heads
        )

        # our patching and embedding from ViT
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print("num_patches: ", num_patches)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # layers

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer_num = idx + 1

            use_xl_memories = layer_num in self.xl_memory_layers
            use_knn_attention = layer_num in memorizing_layers
            xl_max_memories_layer = 0 if not use_xl_memories else xl_max_memories

            if use_knn_attention:
                attn = KNNAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=knn_attn_heads,
                    dropout=attn_dropout,
                    num_retrieved_memories=num_retrieved_memories,
                    xl_max_memories=xl_max_memories_layer,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    xl_max_memories=xl_max_memories_layer,
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        block_wrapper(attn),
                        block_wrapper(
                            FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
                        ),
                    ]
                )
            )

        # memory layer shifting
        # from a little known paper https://arxiv.org/abs/2012.15688

        self.shift_knn_memories_down = shift_knn_memories_down
        self.shift_xl_memories_down = shift_xl_memories_down

        # # to logits

        # self.to_logits = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_tokens)
        # )
        # change logits to mlp out

        self.mlp_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # self.mlp_out = nn.Sequential(
        #     Reduce("b n d -> b d", "mean"),
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes),
        # )

        # knn memories init

        self.knn_mem_kwargs = dict(
            dim=self.dim_head,
            max_memories=self.max_knn_memories,
            multiprocessing=knn_memory_multiprocessing,
        )

    def create_knn_memories(self, *, batch_size):
        return KNNMemoryList.create_memories(
            batch_size=batch_size,
            num_memory_layers=self.num_memory_layers,
            memories_directory=self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(self, **kwargs):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok=True, parents=True)
        lock = FileLock(str(knn_dir / "mutex"))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            self.knn_memories = knn_memories
            yield knn_memories
            knn_memories.cleanup()

    def clear_memory(self, x, token_id):
        """clears the KNN memories based on if the batch row contains the specified token id"""
        """ for auto-clearing KNN memories based on start and end of strings """

        clear_memory = (x == token_id).any(dim=-1)
        batch_indices, _ = clear_memory.nonzero(as_tuple=True)
        batch_indices_to_clear = batch_indices.tolist()

        if len(batch_indices_to_clear) == 0:
            return

        self.knn_memories.clear_memory(batch_indices_to_clear)

    def forward(
        self, x, knn_memories, xl_memories=None,  add_knn_memory=True
    ):

        # print("before embed:",x.shape) # batch token
        x = self.to_patch_embedding(x)
        batch_size, seq_len, *_, device = *x.shape, x.device
        cls_tokens = repeat(self.cls_token, "1 n d -> b n d", b=batch_size)
        # print('after patch embed: ', x.shape)

        x = torch.cat((cls_tokens, x), dim=1)
        # print('after adding cls token: ', x.shape)
        # print(self.pos_embedding[:, :(seq_len+ 1 )].shape)

        x += self.pos_embedding[:, : (seq_len + 1)]
        # x = self.token_emb(x)
        # print("after embed: ", x.shape) # --> batch, tokens, dim

        # validate KNN memories to have enough indices for batch size

        # for memory in knn_memories:
        #     print(memory.num_indices, batch_size)

        assert all(
            [memory.num_indices == batch_size for memory in knn_memories]
        ), f"you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices"

        # if KNN memories are passed in, and researcher wants memories auto-cleared on <sos> token detection
        # do the appropriate logic

        if exists(self.clear_memories_on_sos_token_id):
            self.clear_memory(x, self.clear_memories_on_sos_token_id)

        # handle XL memories

        xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
        assert len(xl_memories) == self.num_xl_memory_layers
        has_xl_memories = len(xl_memories) > 0

        # shifting memories a number of layers down, little known technique shown to enhance memories from Ernie-Doc paper

        if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
            knn_memories = [
                *knn_memories[self.shift_knn_memories_down :],
                *knn_memories[: self.shift_knn_memories_down],
            ]

        if len(xl_memories) > 0 and self.shift_xl_memories_down > 0:
            xl_memories = [
                *xl_memories[self.shift_xl_memories_down :],
                *xl_memories[: self.shift_xl_memories_down],
            ]

        # iterate through the memories in order of the ascending layers that contain KNNAttention

        xl_memories_iter = iter(xl_memories)
        knn_memories_iter = iter(knn_memories)

        # positional bias

        max_context_len = max(
            [
                seq_len,
                *map(
                    lambda t: (t.shape[-3] if exists(t) else 0) + seq_len, xl_memories
                ),
            ]
        )

        rel_pos_bias = self.rel_pos_bias(seq_len, max_context_len, device=device)
        knn_rel_pos_bias = self.knn_rel_pos_bias(
            seq_len, max_context_len, device=device
        )

        # print("rel_pos_bias", rel_pos_bias.shape)
        # print("knn_rel_pos_bias", knn_rel_pos_bias.shape)
        # keep track of new xl memories

        new_xl_memories = [] if has_xl_memories else None

        # go through all layers

        for ind, (attn, ff) in enumerate(self.layers):
            layer_num = ind + 1

            is_memorizing_layer = layer_num in self.memorizing_layers
            is_xl_memory_layer = layer_num in self.xl_memory_layers

            # attn_kwargs = dict(rel_pos_bias = rel_pos_bias if not is_memorizing_layer else knn_rel_pos_bias)
            attn_kwargs = dict()
            if is_memorizing_layer:
                attn_kwargs = {
                    **attn_kwargs,
                    "knn_memory": next(knn_memories_iter),
                    "add_knn_memory": add_knn_memory,
                }

            if is_xl_memory_layer:
                attn_kwargs = {**attn_kwargs, "xl_memory": next(xl_memories_iter)}

            # attention

            x_attn, xl_mem = attn(x, **attn_kwargs)
            
            # Residual
            x = x_attn + x

            # add new XL memories if needed

            if exists(xl_mem):
                new_xl_memories.append(xl_mem)

            # feedforward

            x = x + ff(x)

        # to logits
        # x = x[:, 0]
        
        if self.global_pool:
            x = x[:,1:].mean(dim=1) 
        else:
            x = x[:, 0]
    

        return x
