
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import transformers

local_rank = None

from transformers import Qwen2VLImageProcessorFast
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    freeze_backbone: bool = field(default=False)

    use_cache: bool = field(default=False)


@dataclass
class DataArguments:
    data_names: list[str] = field(default_factory=list)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    data_flatten: bool = field(default=False)
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    max_pixels: int = 50176
    min_pixels: int = 768
    base_interval: int = field(default=2)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None,)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False  #是否启用lora微调
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    resume_from_checkpoint: bool = False