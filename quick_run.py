from omegaconf import DictConfig, OmegaConf
import hydra
from src.utils.util import instantiate
from src.train.hippo_train import HiPPOTrainer


@hydra.main(version_base=None, config_path="src/configs/", config_name="config")
def my_app(cfg):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    task = instantiate(cfg)
    print(task)
    trainer = HiPPOTrainer(task=task)
    trainer.run(epochs=10, batch_size=32)


if __name__ == "__main__":
    my_app()
