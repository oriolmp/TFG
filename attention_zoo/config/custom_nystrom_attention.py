from attrs import define, field
from attention_zoo.config.abstract_attention_config import AbstractAttentionConfig


@define
class AttentionConfig(AbstractAttentionConfig):
    name:str = field(default='custom_nystrom_attention')
    num_samples:int = field(default=128)
    pinv_iterations:int = field(default=6)
    attn_dropout:float = field(default= 0.3)
    ff_dropout:float = field(default=0.3)
    dropout:float = field(default= 0.3)
    residual:bool = field(default= False)
    residual_conv_kernel:int = field(default= 33)
    eps:float = field(default=1e-8)

    symmetrization:bool = field(default= False)
    activation_function:str = field(default= 'relu')
    final_normalization:bool = field(default= True)
    use_uniform_sketching:bool = field(default= False)
    use_sample_perc:bool = field(default= False)
