from pathlib import Path


class Config:
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 10
    models_dir: str = Path("mnist", "models")
