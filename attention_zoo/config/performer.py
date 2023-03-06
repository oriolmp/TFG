from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='performer')
    kernel_type:str = field(default='relu')
    rp_dim:int = field(default=256)
