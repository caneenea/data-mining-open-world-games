# Immersion Mining: Clustering RPG Playstyles (RDR2 / Cyberpunk / Witcher 3 Inspired)

This project demonstrates a **data mining pipeline** that clusters open-world RPG players into **playstyle archetypes**
using **K-Means**, based on telemetry-style features (exploration, combat, side content, dialogue, crafting, etc.).

## Why this matters

Games like **Red Dead Redemption 2**, **Cyberpunk 2077**, and **The Witcher 3** generate large amounts of player behavior data.
Clustering helps designers understand different player types and improve:

- quest pacing and content placement
- difficulty/balance
- UI and fast travel design
- engagement and retention

## Methods

- Synthetic telemetry dataset generation
- Standardization (StandardScaler)
- K-Means clustering (k=3)
- PCA visualization for cluster separation

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```
