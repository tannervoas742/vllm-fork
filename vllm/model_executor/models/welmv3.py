# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Welm model compatible with HuggingFace weights."""
import os
from typing import List, Optional, Tuple, Union, Iterable

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig
from .interfaces import SupportsPP
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.activation import NewGELU, SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from .utils import maybe_prefix
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
import torch.nn.functional as F

from .utils import get_input_mask

is_hpu = current_platform.is_hpu()


class WeLMV3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        hidden_act: str,
        fc1_bias: bool = False,
        fc2_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        if hidden_act not in ["swiglu", "gelu", "gelu_fast"]:
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only swiglu and gelu is supported for now.")
        if hidden_act == "swiglu":
            self.act_fn = SiluAndMul()
            h_to_4h_out_channels = [ffn_hidden_size] * 2
            self.dense_h_to_4h = MergedColumnParallelLinear(
                hidden_size,
                h_to_4h_out_channels,
                bias=fc1_bias,
                quant_config=quant_config,
            )
        else:
            self.act_fn = NewGELU()
            self.dense_h_to_4h = ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,
                bias=fc1_bias,
                quant_config=quant_config,
            )
        self.dense_4h_to_h = RowParallelLinear(ffn_hidden_size,
                                               hidden_size,
                                               bias=fc2_bias,
                                               quant_config=quant_config)

    def forward(self, x):
        gate_up, _ = self.dense_h_to_4h(x)
        x = self.act_fn(gate_up)
        x, _ = self.dense_4h_to_h(x)
        return x


class WeLMV3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        compress: float = 1.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
    ):
        super().__init__()
        self.enable_zero_padding = os.environ.get('VLLM_ZERO_PADDING',
                                                  'false').lower() == 'true'
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.compress = compress

        self.query_key_value = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling={
                "rope_type": "linear",
                "factor": 1 / self.compress
            },
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if (is_hpu and self.enable_zero_padding
                and attn_metadata.seq_lens_tensor is not None):
            valid_len = attn_metadata.seq_lens_tensor
            mask = get_input_mask(hidden_states, valid_len)
            hidden_states = hidden_states * mask.unsqueeze(-1)
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        if (is_hpu and self.enable_zero_padding
                and attn_metadata.seq_lens_tensor is not None):
            attn_output = attn_output * mask.unsqueeze(-1)

        output, _ = self.dense(attn_output)
        return output


#@torch.compile
def welmv3_layer_norm_func(x, residual, normalized_shape, weight, bias, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x += residual
        residual = x.to(orig_dtype)

    x = x.to(orig_dtype)
    x = F.layer_norm(x, normalized_shape, weight, bias, eps)
    if residual is None:
        return x
    else:
        return x, residual


class WeLMV3LayerNorm(LayerNorm):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return welmv3_layer_norm_func(x, residual, self.normalized_shape,
                                      self.weight, self.bias, self.eps)


def get_norm_cls(norm_type: str):
    if norm_type == "rms_norm":
        return RMSNorm
    elif norm_type == "layer_norm":
        return WeLMV3LayerNorm
    raise ValueError(f"Unsupported norm type: {norm_type}")


class WeLMV3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rotary_emb_base", 1000000)
        self.attention = WeLMV3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_kv_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            compress=config.rotary_compress,
            qkv_bias=getattr(config, "qkv_proj_bias", True),
            out_bias=getattr(config, "out_proj_bias", True),
        )
        self.mlp = WeLMV3MLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            fc1_bias=getattr(config, "mlp_fc1_bias", True),
            fc2_bias=getattr(config, "mlp_fc2_bias", True),
        )
        norm_cls = get_norm_cls(getattr(config, 'norm_type', 'layer_norm'))

        self.input_layernorm = norm_cls(config.hidden_size,
                                        eps=config.layernorm_epsilon)
        self.post_attention_layernorm = norm_cls(config.hidden_size,
                                                 eps=config.layernorm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class WeLMV3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.enable_zero_padding = os.environ.get('VLLM_ZERO_PADDING',
                                                  'false').lower() == 'true'
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_in = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            WeLMV3DecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_layers)
        ])
        norm_cls = get_norm_cls(getattr(config, 'norm_type', 'layer_norm'))
        self.final_layer_norm = norm_cls(config.hidden_size,
                                         eps=config.layernorm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_in(input_ids)
        residual = None

        if is_hpu:
            if (self.enable_zero_padding
                    and attn_metadata.seq_lens_tensor is not None):
                valid_len = attn_metadata.seq_lens_tensor
                mask = get_input_mask(hidden_states, valid_len)
                hidden_states = hidden_states * mask.unsqueeze(-1)

            import habana_frameworks.torch as htorch
            htorch.core.mark_step()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
            if is_hpu:
                htorch.core.mark_step()
        hidden_states, _ = self.final_layer_norm(hidden_states, residual)
        return hidden_states


class WeLMV3ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.lm = WeLMV3Model(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "model"))
        self.embed_out = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.sampler = Sampler()
        self.swiglu = config.hidden_act == "swiglu"
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size, logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.lm(input_ids, positions, kv_caches, attn_metadata,
                                inputs_embeds)
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.embed_out, hidden_states,
                                       sampling_metadata)
        return logits

    # welm中部分dense参数的layout和vllm要求的不一致，需要在加载权重时进行调整
    # 于是有了reshape_query_key_value / reshape_key_value_dense / reshape_dense_h_to_4h
    def reshape_query_key_value(self, output_dim: int,
                                loaded_weight: torch.Tensor,
                                is_fp8_scale: bool):
        if is_fp8_scale:
            return loaded_weight
        loaded_weight_shape = loaded_weight.shape
        loaded_weight = loaded_weight.view(
            loaded_weight_shape[:output_dim] +
            (self.config.num_attention_heads, 3, -1) +
            loaded_weight_shape[output_dim + 1:])
        loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)
        loaded_weight = loaded_weight.reshape(loaded_weight_shape)
        return loaded_weight

    def reshape_key_value_dense(self, output_dim: int,
                                loaded_weight: torch.Tensor,
                                is_fp8_scale: bool):
        if is_fp8_scale:
            return [loaded_weight, loaded_weight]
        loaded_weight_shape = loaded_weight.shape
        loaded_weight = loaded_weight.view(loaded_weight_shape[:output_dim] +
                                           (self.config.num_kv_heads, 2, -1) +
                                           loaded_weight_shape[output_dim +
                                                               1:])
        loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)
        loaded_weight = loaded_weight.reshape(loaded_weight_shape)
        loaded_weight = loaded_weight.chunk(2, dim=0)
        return loaded_weight

    def reshape_dense_h_to_4h(self, output_dim: int,
                              loaded_weight: torch.Tensor, is_fp8_scale: bool):
        if is_fp8_scale:
            if self.swiglu:
                loaded_weight = loaded_weight.repeat(2)
            return loaded_weight
        if self.swiglu:
            loaded_weight = loaded_weight.chunk(2, dim=output_dim)
            loaded_weight = torch.concat(loaded_weight[-1::-1], dim=output_dim)
        return loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("query_key_value", "query_key_value", None),
            ("query_key_value", "key_value_dense", ["k", "v"]),
            ("query_key_value", "query_dense", "q"),
            ("dense_h_to_4h", "dense_h_to_4h", None),
        ]

        weights_to_reshape = {
            "query_key_value": self.reshape_query_key_value,
            "key_value_dense": self.reshape_key_value_dense,
            "dense_h_to_4h": self.reshape_dense_h_to_4h,
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".g_idx"):
                    continue
                param = params_dict[name]

                for key, reshape_func in weights_to_reshape.items():
                    if key not in weight_name:
                        continue
                    output_dim = getattr(param, "output_dim", None)
                    is_fp8_scale = getattr(param, "needs_scalar_to_array",
                                           False) is True
                    loaded_weight = reshape_func(output_dim, loaded_weight,
                                                 is_fp8_scale)

                weight_loader = param.weight_loader
                if isinstance(shard_id, list):
                    for i, shard in enumerate(shard_id):
                        weight_loader(param, loaded_weight[i], shard)
                elif shard_id:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                if self.config.tie_word_embeddings and "lm.embed_in" in name:
                    share_name = name.replace("lm.embed_in", "embed_out")
                    param = params_dict[share_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            if is_hpu:
                torch.hpu.synchronize()
