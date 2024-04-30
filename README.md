# TargetGuardingRL
This is an implementation of the target guarding differential game for CS394R.

Using DQN, the agent (pursuer) learns to defend a target area by intercepting or chasing away an attacking evader.

## Installation
```bash
git clone https://github.com/adiaz6/TargetGuardingRL.git
```

## Contents 

```bash
environment/
```
Holds world, agent, and sprite models

```bash
dqn.py
```
DQN algorithm (from https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py)

```bash
training.py
```
For training

```bash
comparison.py
```
For evaluation

```bash
model_baseline.pt
```
Trained policy with baseline reward

```bash
model_phase1.pt
```
Trained policy with Phase I reward

```bash
model_phase2.pt
```
Trained policy with Phase II reward

## Usage
### Training
```bash
python3 training.py
```
This will begin training with DQN.

### Evaluation
```bash
python3 comparison.py
```
This will evaluate the policies and create plots for reward and success rate.
