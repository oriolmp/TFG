from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='nystromformer')
    num_landmarks:int = field(default=64)
    pinv_iterations:int = field(default=6)
    attn_values_residual:bool = field(default=True)
    attn_values_residual_conv_kernel:int = field(default=33)
    attn_dropout:float = field(default=0.3)
    ff_dropout:float = field(default=0.3)
    dropout:float = field(default=0.3)
    residual:bool = field(default=True)
    residual_conv_kernel:int = field(default=33)
    eps:float = field(default=1e-8)
