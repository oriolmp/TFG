from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='custom_full_attention')
    symmetrization:bool = field(default=False)
    activation_function:str = field(default='linear')
    final_normalization:bool = field(default=False)
