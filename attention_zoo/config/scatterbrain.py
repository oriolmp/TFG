from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='scatterbrain')
    local_context:int = field(default=50)
    ortho_scaling:int = field(default=0)
    causal:bool = field(default=False)
    softmax_temp:float = field(default=None)
    attention_dropout:float = field(default=0.0)
    softmax_eps:float = field(default=0.0)
