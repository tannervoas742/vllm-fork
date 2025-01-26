# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only ChatGLM model compatible with THUDM weights."""
import os
from argparse import Namespace
from array import array
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.nn import LayerNorm

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4_vision_encoder import EVA2CLIPModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.platforms import current_platform
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors,
                           SequenceData)
from vllm.transformers_utils.configs import ChatGLMConfig

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import get_input_mask

logger = init_logger(__name__)


def calculate_image_placeholder(vision_config):
    return (vision_config["image_size"] // vision_config["patch_size"] // 2)**2


def mm_input_mapper_for_glmv(
    ctx: InputContext,
    data: LLMInputs,
) -> Dict:
    return MultiModalKwargs()


def get_max_glmv_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(ChatGLMConfig)

    vision_config = getattr(hf_config, 'vision_config', None)
    if vision_config is None:
        return 1
    elif isinstance(vision_config, dict):
        return calculate_image_placeholder(vision_config)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def dummy_data_for_glmv(
    ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]
) -> Tuple[SequenceData, Optional[MultiModalDataDict]]:
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = getattr(hf_config, 'vision_config', None)

    if vision_config is None:
        token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [0] * seq_len)
        seq_data = SequenceData(token_ids)
        return seq_data, None
    elif isinstance(vision_config, dict):
        image_size = vision_config["image_size"]
        image_placeholder_length = calculate_image_placeholder(vision_config)
        token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [hf_config.boi_token_id] +
                          [0] * image_placeholder_length +
                          [hf_config.eoi_token_id])
        token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                           [0] * (seq_len - image_placeholder_length - 2))
        seq_data = SequenceData(token_ids)

        mm_data = {
            "image": Image.new("RGB", (image_size, image_size), color=0)
        }

        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def find_all_positions(input_ids: List[int], target: int) -> List[int]:
    return [index for index, value in enumerate(input_ids) if value == target]


def input_processor_for_glmv(ctx: InputContext, llm_inputs: LLMInputs):
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = getattr(hf_config, 'vision_config', None)
    if vision_config is None:
        return llm_inputs
    elif isinstance(vision_config, dict):
        image_placeholder_length = calculate_image_placeholder(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)
    tokenizer = cached_get_tokenizer(
        ctx.model_config.model,
        trust_remote_code=ctx.model_config.trust_remote_code)

    try:
        raw_batch_data = tokenizer.apply_chat_template(
            conversation=[{
                "role": "user",
                "image": llm_inputs['multi_modal_data']["image"],
                "content": llm_inputs['prompt']
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True).data

    except Exception:
        logger.error("Failed to process content (%s)", llm_inputs['prompt'])
        raise
    pixel_values = raw_batch_data['images'].tolist()

    input_ids = raw_batch_data['input_ids'].tolist()
    position_ids = raw_batch_data['position_ids'].tolist()
    attention_mask = raw_batch_data['attention_mask'].tolist()
    img_idx = []
    batch_size = len(input_ids)
    for i in range(batch_size):
        boi_token_pos, eoi_token_pos = input_ids[i].index(
            hf_config.boi_token_id), input_ids[i].index(hf_config.eoi_token_id)
        assert eoi_token_pos - boi_token_pos == 2
        new_input_ids = input_ids[i][:boi_token_pos + 1] + [
            input_ids[i][-1]
        ] * image_placeholder_length + input_ids[i][eoi_token_pos:]
        new_position_ids = position_ids[i][:boi_token_pos + 1] + [
            position_ids[i][boi_token_pos + 1]
        ] * image_placeholder_length + position_ids[i][eoi_token_pos:]
        new_attention_mask = attention_mask[i][:boi_token_pos + 1] + [
            1
        ] * image_placeholder_length + attention_mask[i][eoi_token_pos:]
        new_image_idx = list(
            range(boi_token_pos, boi_token_pos + image_placeholder_length + 2))

        input_ids[i] = new_input_ids
        position_ids[i] = new_position_ids
        attention_mask[i] = new_attention_mask
        img_idx.append(new_image_idx)
    multi_modal_data = llm_inputs.get("multi_modal_data")
    multi_modal_data["img_idx"] = torch.tensor(img_idx, dtype=torch.long)
    multi_modal_data["img_position_ids"] = torch.tensor(position_ids,
                                                        dtype=torch.long)
    multi_modal_data["pixel_values"] = torch.tensor(pixel_values,
                                                    dtype=torch.bfloat16)
    llm_inputs['multi_modal_data'] = multi_modal_data
    llm_inputs["prompt_token_ids"] = torch.tensor(
        input_ids, dtype=torch.int32).reshape(-1)
    llm_inputs["attention_mask"] = torch.tensor(attention_mask,
                                                dtype=torch.long)

    return llm_inputs


is_hpu = current_platform.is_hpu()

try:
    from htorch.hpex.kernels import RotaryPosEmbeddingHelperV3 as FusedRoPE
except ImportError:
    print("Cannot import Fused Rope from Habana Torch")
    FusedRoPE = None


class GLMAttention(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.enable_zero_padding = os.environ.get('VLLM_ZERO_PADDING',
                                                  'false').lower() == 'true'
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )
        self.vision_config_flag = getattr(config, 'vision_config', None)
        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim // 2,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=False,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
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
        q, k = self.rotary_emb(position_ids, q, k)

        context_layer = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size,
                                               eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = GLMAttention(config, cache_config, quant_config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.enable_zero_padding = os.environ.get('VLLM_ZERO_PADDING',
                                                  'false').lower() == 'true'

        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList([
            GLMBlock(config, cache_config, quant_config)
            for i in range(self.num_layers)
        ])

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon)
        self.vision_config_flag = getattr(config, 'vision_config', None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        if (is_hpu and (self.vision_config_flag is None)
                and self.enable_zero_padding
                and (attn_metadata.seq_lens_tensor is not None)):
            valid_len = attn_metadata.seq_lens_tensor
            mask = get_input_mask(hidden_states, valid_len)
            hidden_states = hidden_states * mask.unsqueeze(-1)

        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embedding = VocabParallelEmbedding(config.padded_vocab_size,
                                                config.hidden_size)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        self.encoder = GLMTransformer(config, cache_config, quant_config)

        self.output_layer = ParallelLMHead(config.padded_vocab_size,
                                           config.hidden_size,
                                           quant_config=quant_config)

        self.vision_config_flag = getattr(config, 'vision_config', None)
        if self.vision_config_flag is not None:
            self.config = config
            self.vision_config = Namespace(**config.vision_config)
            self.vision = EVA2CLIPModel(self.config, quant_config)
        else:
            self.vision = None

    def _parse_and_validate_image_input(self, pixel_values) -> torch.Tensor:
        if pixel_values is not None and self.vision is not None:
            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.ndim > 2:
                    pixel_values = torch.concat(list(pixel_values))
            elif isinstance(pixel_values, list):
                return torch.concat(pixel_values)
            else:
                raise TypeError("""pixel_values must be a torch.Tensor 
                    or a list of torch.Tensor
                    """)
        return pixel_values

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        pixel_values: Optional[torch.Tensor] = None,
        img_idx: Optional[torch.LongTensor] = None,
        img_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids)

        if self.vision_config_flag is not None:
            pixel_values = self._parse_and_validate_image_input(pixel_values)
        else:
            pixel_values = None

        if pixel_values is not None and self.vision_config_flag is not None:
            image_embeds = self.vision(pixel_values)

        if pixel_values is not None and self.vision is not None:            
            batch_size, seq_length, hidden_size = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(-1,hidden_size)
            image_embeds = image_embeds.reshape(-1,hidden_size)
            img_idx = img_idx.reshape(-1)
            inputs_embeds.index_copy_(0, img_idx, image_embeds)
            inputs_embeds = inputs_embeds.reshape(batch_size, seq_length, hidden_size)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        return hidden_states


def input_imageidx_mapper_for_glmv(ctx: InputContext,
                                   data: object) -> MultiModalKwargs:
    img_idx = data
    return MultiModalKwargs({"img_idx": img_idx})


def input_imgpositionids_mapper_for_glmv(ctx: InputContext,
                                         data: object) -> MultiModalKwargs:
    input_positions = data
    return MultiModalKwargs({"img_position_ids": input_positions})


def input_pixelValues_mapper_for_glmv(ctx: InputContext,
                                      data: object) -> MultiModalKwargs:
    pixel_values = data
    return MultiModalKwargs({"pixel_values": pixel_values})


@MULTIMODAL_REGISTRY.register_input_mapper("img_position_ids",
                                           input_imgpositionids_mapper_for_glmv
                                           )
@MULTIMODAL_REGISTRY.register_input_mapper("img_idx",
                                           input_imageidx_mapper_for_glmv)
@MULTIMODAL_REGISTRY.register_input_mapper("pixel_values",
                                           input_pixelValues_mapper_for_glmv)
@MULTIMODAL_REGISTRY.register_image_input_mapper(mm_input_mapper_for_glmv)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_glmv_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_glmv)
@INPUT_REGISTRY.register_input_processor(input_processor_for_glmv)
class ChatGLMForCausalLM(nn.Module, SupportsLoRA, SupportsPP,
                         SupportsMultiModal):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"]
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
    ):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.max_position_embeddings = getattr(self.config,
                                               "max_sequence_length", 8192)
        self.transformer = ChatGLMModel(vllm_config=vllm_config)
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(self.config.padded_vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        pixel_values: Optional[torch.Tensor] = None,
        img_idx: Optional[torch.LongTensor] = None,
        img_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, pixel_values, img_idx,
                                         img_position_ids)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Merge two ColumnParallelLinear into one MergedColumnParallelLinear
        merged_weights_dict: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
            "transformer.vision.linear_proj.merged_proj.weight": {
                "transformer.vision.linear_proj.gate_proj.weight": None,
                "transformer.vision.linear_proj.dense_h_to_4h.weight": None,
            }
        }

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            is_weight_to_be_merge = False
            for _, merged_weight_dict in merged_weights_dict.items():
                if name in merged_weight_dict:
                    assert merged_weight_dict[name] is None
                    merged_weight_dict[name] = loaded_weight
                    is_weight_to_be_merge = True
            if is_weight_to_be_merge:
                continue

            if "rotary_pos_emb.inv_freq" in name:
                continue
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        for combined_name, merged_weight_dict in merged_weights_dict.items():
            if combined_name in params_dict:
                param = params_dict[combined_name]
                combined_weight = torch.cat(list(merged_weight_dict.values()),
                                            dim=0)
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, combined_weight)
