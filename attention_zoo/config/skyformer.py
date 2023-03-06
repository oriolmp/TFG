from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='skyformer')
    pinv_iterations:int = field(default=6)
    num_feats:int = field(default=128)
    attn_values_residual:bool = field(default=True)
    attn_values_residual_conv_kernel:int = field(default=33)
    attn_dropout:float = field(default=0.3)
    ff_dropout:float = field(default=0.3)
    dropout:float = field(default=0.3)
    no_projection:bool = field(default=False)
    accumulation:int = field(default=1)
    residual:bool = field(default=False)