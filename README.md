# Bionic Hand – Isaac Lab Environments

This repository contains custom Isaac Lab tasks for training and evaluating reinforcement learning (RL) agents on different variants of a bionic hand model. The setup is based on **Isaac Sim 5.0.0** and **Isaac Lab 2.0**, running inside a Docker container.

---

## Setup

Clone the repository:

```bash
git clone [https://github.com/<your-user>/<your-repo>.git](https://github.com/YuziIV/BionicHand.git)
cd BionicHand/IsaacLab/docker
```

---

## Docker Container

From inside the `docker` folder:

```bash
# Start the container
sudo ./container.py start

# Enter the container
sudo ./container.py enter
```

Once inside the container, you can launch Isaac Lab via the helper script.

---

## Training an RL Agent

To start training, run:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Repose-Cube-BionicHand-Direct-v0 \
    --algorithm PPO \
    --headless
```

This command will train the **8-DoF bionic hand** variant and log results under:

```
logs/skrl/bionic_8x_hand/<timestamp>_ppo_torch/
```

To train the **7-DoF original bionic hand**, use:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Repose-Cube-OrigBionicHand-Direct-v0 \
    --algorithm PPO \
    --headless
```

Logs will be saved under:

```
logs/skrl/bionic_hand/<timestamp>_ppo_torch/
```

---

## Playing / Evaluating a Policy

After training, you can play back or evaluate the trained agent. Example for the 8-DoF hand:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-Repose-Cube-BionicHand-Direct-v0 \
    --algorithm PPO \
    --checkpoint /workspace/isaaclab/logs/skrl/bionic_8x_hand/2025-09-25_17-21-35_ppo_torch/checkpoints/best_agent.pt \
    --num_envs 9
```

- Replace the `--checkpoint` path with the one inside your `logs/skrl/.../checkpoints/` folder.
- `--num_envs` sets the number of parallel environments to run during evaluation.

For the 7-DoF hand, just change the task:

```bash
--task Isaac-Repose-Cube-OrigBionicHand-Direct-v0
```

---

## Log Structure

- **8-DoF Bionic Hand:**  
  `logs/skrl/bionic_8x_hand/<timestamp>_ppo_torch/`

- **7-DoF Bionic Hand:**  
  `logs/skrl/bionic_hand/<timestamp>_ppo_torch/`

Each run folder contains:
- `params/` – saved configs (`env.yaml`, `agent.yaml`)
- `checkpoints/` – agent weights
- `videos/` (optional) – recorded rollouts
- `evaluation.csv` (if evaluated with `play.py`)
