# Experiments Roadmap

AI/ML experiments using SuperMarioBros gameplay data recorded with gymrec.
Assumes a frame-row dataset of dozens of episodes recorded via
`uv run gymrec record SuperMarioBros-Nes-v0 --storage images`.

Each experiment builds on the previous ones. Follow the suggested learning path at the bottom.

---

## Level 1 — Data Exploration

**1.1 Dataset Explorer Notebook**
Load the dataset, render frames as a video, plot reward curves per episode, visualize action
distribution (which NES buttons get pressed most), show info fields over time. Pure
pandas/matplotlib — zero ML, but essential to understand what you're working with.

**1.2 Frame Deduplication & Quality**
Identify near-duplicate frames (consecutive static frames when Mario dies/pauses). Teaches
hashing and perceptual similarity.

---

## Level 2 — Embeddings & Representation

**2.1 Pretrained CNN Embeddings**
Pass all frames through a frozen ResNet/EfficientNet, extract the penultimate layer as a
512-d vector per frame. No training needed. Teaches transfer learning fundamentals.

**2.2 Embedding Space Visualization**
Run UMAP or t-SNE on those embeddings, color points by reward or episode step. You'll see
game states cluster naturally (underground levels vs overworld vs death screens). Teaches
dimensionality reduction.

**2.3 CLIP Zero-Shot Queries**
Use CLIP to ask natural language questions about frames: *"Is Mario near an enemy?"*,
*"Is Mario underground?"*. No training, pure zero-shot. Teaches multimodal models and
prompt-based retrieval.

**2.4 Nearest Neighbor Frame Search**
Given any frame, find the top-k most similar frames across all episodes using embedding
distance. Useful as a debugging tool for everything that follows.

---

## Level 3 — Classification

**3.1 VLM Auto-Labeling**
Use Claude API to caption every frame with structured labels (Mario alive/dead, powerup
status, enemy count, level section). Creates a rich annotated dataset for free. Teaches
multimodal APIs and dataset augmentation.

**3.2 Game State Classifier**
Train a small CNN to classify frames into states (overworld/underground/castle/death) using
the VLM labels as ground truth. Teaches image classification, train/val splits, confusion
matrices.

**3.3 Event Detection**
Classify frames as "Mario collected coin", "Mario took damage", "Mario jumped on enemy".
These are sparse events — teaches class imbalance handling.

---

## Level 4 — Autoencoders & Generation

**4.1 Convolutional Autoencoder**
Train an encoder-decoder to compress 240×256 frames to a small latent vector and reconstruct
them. Teaches unsupervised representation learning, conv architectures, reconstruction loss.

**4.2 Variational Autoencoder (VAE)**
Same idea but with a probabilistic latent space — you can sample new game frames. Teaches
KL divergence, the reparameterization trick, generative models.

**4.3 Latent Space Arithmetic**
Interpolate between two frames in latent space and decode the path — watch the game "morph"
between two states. Pure fun, teaches what smooth latent spaces mean.

---

## Level 5 — Sequence Modeling

**5.1 Action Sequence Modeling**
Train an LSTM or small Transformer on sequences of actions. Given the last N button presses,
predict the next one. Teaches sequence modeling basics.

**5.2 (Frame, Action) Sequence Modeling**
Extend 5.1 to condition on frames too — given the last N (frame, action) pairs, predict the
next action. The simplest form of a policy model with memory.

**5.3 Reward Prediction**
Train a model to predict the reward at each step from the current frame. Teaches value
estimation — the foundation of every RL critic network.

---

## Level 6 — Imitation Learning

**6.1 Behavioral Cloning (BC)**
Train a CNN: frame → action vector. Supervised learning, cross-entropy loss on the
MultiBinary NES buttons. Evaluate by running the agent live in stable-retro and watching it
play. The payoff here is huge.

**6.2 BC with Temporal Context**
Feed a stack of the last 4 frames instead of a single frame (classic DQN trick). Compare how
much temporal context improves the clone's performance.

**6.3 DAgger (Dataset Aggregation)**
Iterative imitation learning: run the BC agent, record where it drifts from human behavior,
add those correction examples to training data, retrain. Fixes the distribution shift problem
in vanilla BC. Teaches interactive/online learning.

---

## Level 7 — Reinforcement Learning

**7.1 PPO from Scratch (Baseline)**
Train a PPO agent on Mario with no prior data — just raw RL. Use this as a baseline to
measure how much your human data helps. Teaches RL fundamentals: policy gradients, value
functions, GAE.

**7.2 BC Warm-Start → PPO**
Initialize the PPO policy with your BC weights, then fine-tune with RL. Compare convergence
speed and final performance against the from-scratch baseline. This is the classic
"imitation + RL" paradigm.

**7.3 Offline RL (CQL or IQL)**
Train entirely from your recorded dataset, no live environment interaction. Uses Conservative
Q-Learning or Implicit Q-Learning to be pessimistic about out-of-distribution actions.
Teaches offline/batch RL — a huge research area.

**7.4 Reward Shaping**
Use your learned reward predictor from 5.3 as a shaped reward signal to make RL learn
faster. Teaches reward engineering.

---

## Level 8 — Advanced / Modern

**8.1 World Model (RSSM)**
Train a recurrent state-space model that learns to predict next frames in latent space given
actions. The agent can then "imagine" future trajectories without touching the real
environment (Dreamer-style). Teaches model-based RL.

**8.2 Decision Transformer**
Model the entire trajectory as a token sequence: `(return, state, action, return, state,
action, ...)`. At inference time, condition on a high desired return to get a good agent.
Teaches the sequence modeling approach to RL — a very elegant paper from 2021.

**8.3 Contrastive Learning (SimCLR)**
Learn frame representations by training the model to agree on augmented views of the same
frame and disagree on different frames. No labels needed. Teaches self-supervised learning.

**8.4 Fine-tune a Small VLM**
Use your VLM-captioned frames (from 3.1) to fine-tune a small vision-language model
(SmolVLM, LLaVA-1.5) to answer questions about Mario game states. Teaches VLM fine-tuning
and instruction tuning.

---

## Suggested Learning Path

```
1.1 → 2.1 → 2.2 → 3.1 → 3.2   ← solid foundations
     ↓
4.1 → 4.2                        ← generative models detour (optional)
     ↓
5.1 → 5.2 → 5.3                 ← sequences & value estimation
     ↓
6.1 → 6.2 → 6.3                 ← imitation learning
     ↓
7.1 → 7.2 → 7.3                 ← reinforcement learning
     ↓
8.2 (Decision Transformer)       ← crown jewel
```

The **crown jewel** is 8.2 — a Decision Transformer conditioned on return-to-go that you
trained entirely from your own Mario gameplay data.
