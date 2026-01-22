from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SynthConfig:
    n_players: int = 1200
    seed: int = 42

def generate_synthetic_players(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # 3 latent playstyles: Explorer, Combat, Narrative/Completion-ish
    # We'll mix them so it looks realistic.
    style = rng.choice([0, 1, 2], size=cfg.n_players, p=[0.4, 0.35, 0.25])

    # Core telemetry-like features
    exploration_hours = rng.normal(loc=[80, 35, 55], scale=[18, 12, 15])[style]
    combat_encounters = rng.normal(loc=[220, 420, 260], scale=[60, 90, 70])[style]
    side_quests = rng.normal(loc=[55, 22, 65], scale=[12, 10, 14])[style]
    fast_travel_rate = rng.normal(loc=[0.25, 0.55, 0.35], scale=[0.10, 0.12, 0.10])[style]
    dialogue_choices = rng.normal(loc=[140, 85, 170], scale=[25, 20, 30])[style]

    # “Immersion proxy” features (not “true” immersion, but measurable behaviors)
    photo_mode_uses = rng.poisson(lam=[18, 4, 10])[style]
    crafting_actions = rng.normal(loc=[120, 80, 140], scale=[35, 30, 40])[style]
    stealth_ratio = np.clip(rng.normal(loc=[0.35, 0.20, 0.30], scale=[0.12, 0.10, 0.10])[style], 0, 1)

    df = pd.DataFrame({
        "exploration_hours": np.clip(exploration_hours, 1, None),
        "combat_encounters": np.clip(combat_encounters, 1, None),
        "side_quests": np.clip(side_quests, 0, None),
        "fast_travel_rate": np.clip(fast_travel_rate, 0, 1),
        "dialogue_choices": np.clip(dialogue_choices, 0, None),
        "photo_mode_uses": np.clip(photo_mode_uses, 0, None),
        "crafting_actions": np.clip(crafting_actions, 0, None),
        "stealth_ratio": stealth_ratio,
    })

    return df
