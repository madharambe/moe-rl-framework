from dataclasses import dataclass

@dataclass
class NanoGPTConfig:
    model_type: str = "gpt2-micro"

    vocab_size: int = 10000
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True

    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    min_lr: float = 1e-5

    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_iters: int = 5000

    device: str = "cuda"

    dataset: str = "code_corpus"
    data_dir: str = "data/processed"
    out_dir: str = "out/code_gpt"
    log_interval: int = 10
    eval_interval: int = 100
    eval_iters: int = 50
    save_interval: int = 200


@dataclass
class CodeGPTConfig(NanoGPTConfig):
    model_type: str = "code-gpt"
    vocab_size: int = 10000
    block_size: int = 768
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.2
    bias: bool = True

    learning_rate: float = 3e-4
    max_iters: int = 250
    warmup_iters: int = 500
    lr_decay_iters: int = 10000

    curriculum: bool = True
    multi_token_prediction: bool = False
