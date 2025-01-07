import os
from typing import Optional
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Iterable

import torch
from torch import nn

from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from .interfaces import SupportsPP
from vllm.distributed import get_pp_group
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from .utils import PPMissingLayer, maybe_prefix
from .utils import is_pp_missing_parameter
from .utils import make_layers
from .utils import get_input_mask
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.platforms import current_platform


class WeLMMoeMLP(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True,
                 prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class WeLMMoeSparseMoEBlock(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        assert self.tp_size <= config.num_experts

        self.experts = FusedMoE(num_experts=config.num_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=False,
                                renormalize=config.norm_topk_prob,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.shared_expert = None
        if config.num_shared_experts is not None:
            self.shared_expert = WeLMMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
            )

        self.shared_expert_gate = None
        if config.has_shared_expert_gate:
            self.shared_expert_gate = nn.Linear(config.hidden_size,
                                                1,
                                                bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = nn.functional.sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
        if shared_output is not None:
            final_hidden_states += shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(bs, num_tokens, hidden_dim)


class WeLMMoeAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000.,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 compress: float = 1.0,
                 max_position_embeddings: int = 8192,
                 qkv_proj_bias=True,
                 o_proj_bias=False,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
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
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj_bias = qkv_proj_bias
        self.o_proj_bias = o_proj_bias
        self.compress = compress

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=self.qkv_proj_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=self.o_proj_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling is None:
            rope_scaling = {"rope_type": "linear", "factor": 1 / self.compress}
        else:
            assert self.compress == 1.0, "Compress must be 1.0 for custom rope scaling."

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata: AttentionMetadata):
        if (current_platform.is_hpu() and self.enable_zero_padding
                and attn_metadata.seq_lens_tensor is not None):
            valid_len = attn_metadata.seq_lens_tensor
            mask = get_input_mask(hidden_states, valid_len)
            hidden_states = hidden_states * mask.unsqueeze(-1)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)

        if (current_platform.is_hpu() and self.enable_zero_padding
                and attn_metadata.seq_lens_tensor is not None):
            attn_output = attn_output * mask.unsqueeze(-1)

        return output


class WeLMMoeDecoderLayer(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            #layer_idx: int,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        layer_idx = int(prefix.split(".")[-1])
        self.self_attn = WeLMMoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            compress=config.rotary_compress,
            max_position_embeddings=max_position_embeddings,
            qkv_proj_bias=config.qkv_proj_bias,
            o_proj_bias=config.out_proj_bias,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers and config.num_experts > 0
                and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = WeLMMoeSparseMoEBlock(config=config,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.mlp")
        else:
            self.mlp = WeLMMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata: AttentionMetadata,
                residual: Optional[torch.Tensor]) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class WeLMMoeModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.enable_zero_padding = os.environ.get('VLLM_ZERO_PADDING',
                                                  'false').lower() == 'true'

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: WeLMMoeDecoderLayer(config,
                                               prefix,
                                               cache_config=cache_config,
                                               quant_config=quant_config),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor,
            kv_caches: List[torch.Tensor], attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors]
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            residual = intermediate_tensors["residual"]
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if current_platform.is_hpu():
            if (self.enable_zero_padding
                    and attn_metadata.seq_lens_tensor is not None):
                valid_len = attn_metadata.seq_lens_tensor
                mask = get_input_mask(hidden_states, valid_len)
                hidden_states = hidden_states * mask.unsqueeze(-1)
            import habana_frameworks.torch as htorch
            htorch.core.mark_step()
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)
            if current_platform.is_hpu():
                htorch.core.mark_step()
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class WeLMMoeForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = WeLMMoeModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors)

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def sample(self, logits: Optional[torch.Tensor],
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in weights:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts." in name and name not in params_dict:
                    continue

                name = name.replace(weight_name, param_name)
                if (name.endswith(".bias")
                        or name.endswith("_bias")) and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if (name.endswith(".bias") or name.endswith("_bias")
                        ) and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if (name.endswith(".bias") or name.endswith("_bias")
                        ) and name not in params_dict:
                        continue

                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_size")
                        name = remapped_kv_scale_name

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            if current_platform.is_hpu():
                torch.hpu.synchronize()
