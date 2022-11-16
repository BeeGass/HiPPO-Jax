from omegaconf import DictConfig, OmegaConf
import hydra
from src.models.rnn.train import recurrent_train

if __name__ == "__main__":
    recurrent_train()
