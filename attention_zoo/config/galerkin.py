from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='galerkin')
