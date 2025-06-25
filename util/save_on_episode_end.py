import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnEpisodeEnd(BaseCallback):
    """
    Save the model at the end of every episode.
    """
    def __init__(self, save_dir: str, model_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Loop through all envs in VecEnv
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                fname = os.path.join(self.save_dir, f"snapshot-{self.model_name}")
                self.model.save(fname)

        return True