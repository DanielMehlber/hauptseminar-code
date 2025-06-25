from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ConvergenceCallback(BaseCallback):
    def __init__(self, window=15, tol=200):
        super().__init__()
        self.rewards = []
        self.episode_count = 0
        self.window = window
        self.tol = tol

    def _on_step(self) -> bool:
        # Every step, self.locals['infos'] gets gym info dicts
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                r = info['episode']['r']
                self.rewards.append(r)
                self.episode_count += 1
                if len(self.rewards) >= self.window:
                    last = self.rewards[-self.window:]
                    print(f"Reward after {self.episode_count} episodes: {r} (std {np.std(last)})")
                    if np.std(last) < self.tol:
                        return False
        return True