from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='fastformer')
    alternating:bool = field(default= True)
    epsilon:float = field(default= 0.0)
    steps:int = field(default= 1000)
    ff_mult:int = field(default= 1)
    absolute_pos_emb:bool = field(default= False)
    dropout:float = field(default= 0.1)
    use_rotary_emb:bool = field(default= False)
    pos_emb:bool = field(default= False)
