# RL‑Tumour‑Localisation

Open‑source implementation that accompanies the MSc thesis  
*“Accelerating Tumour Localisation in 3‑D MRI with Reinforcement Learning”*.

| File | Purpose |
|------|---------|
| `main.py` | Re‑runs training (30 M steps PPO) **or** evaluates the pre‑trained model. |
| `requirements.txt` | Minimal Python packages (tested with Python 3.7 ). |
| `cc.zip` | Pre‑trained weights (uploaded under Releases v1.0). |

## Quick start
```bash
pip install -r requirements.txt
python main.py --timesteps 3e7
