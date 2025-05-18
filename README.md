## Probabilistic Graphical Models on Race Event Data

This project was developed as a final assignment for the course **"Probabilistic Graphical Models"**, part of the Master's program in Machine Learning. It explores advanced modeling techniques such as Bayesian Networks, Gibbs Sampling, and Variational Autoencoders using real-world data from endurance race events.

---

## **Dataset Description**

The project uses two datasets provided through the course:

### **`share-races.csv`**  
Describes various sport events and races (4,621 rows):
- An event typically occurs over multiple years and includes several races.
- Each race records:
  - Number of participants (`nb_started`)
  - Proportion of finishers (`prop_dropped`, scaled by 100000)
  - Finish time of the first participant
  - Start date, latitude, total distance (`total_dist`)
  - Difficulty (`total_dpos`), and altitude stats (min, mean, max)

### **`share-splits.csv`**  
Contains performance data for checkpoints ("splits") during races (2,499,071 rows):
- Each entry includes:
  - Race ID, participant ID
  - Time to reach the checkpoint (`time`)
  - Distance from the start and difficulty (`dpos`) at that point

**Note**: Missing values are represented as `-999`.

---

## Project Pipeline

The pipeline includes multiple stages, each applying concepts from probabilistic graphical modeling and machine learning.

---

### 🔹 1. Data Preprocessing & Exploration
- Replaced missing values (`-999`) with `NaN`, then imputed using **median values**.
- Selected and standardized features like:
  - `total_dist`, `total_dpos`, `altitudes`, `prop_dropped`
- Plotted distributions and correlations to understand feature behavior.

---

### 🔹 2. Clustering with Gaussian Mixture Model (GMM)
- Applied **GMM** on race-level features: `total_dist`, `total_dpos`, `mean_alt`.
- Used **Bayesian Information Criterion (BIC)** to select the optimal number of clusters.
- Assigned each race to a cluster representing different race types (e.g., short-easy, long-hard).

---

### 🔹 3. Bayesian Network Modeling
- Built a **Bayesian Network** using `pgmpy`, connecting the cluster label to a derived variable `race_duration_category` (short / medium / long).
- Trained the network using **Maximum Likelihood Estimation**.
- Enabled probabilistic inference to compute:
  > P(duration = short | cluster = 1)

---

### 🔹 4. Inference with Gibbs Sampling
- Performed **Gibbs Sampling** on the Bayesian Network to generate samples from the posterior distribution.
- This enabled:
  - Empirical estimation of conditional probabilities
  - Quantification of **aleatoric uncertainty** (uncertainty due to data randomness)

---

### 🔹 5. Top-K Sampling & Uncertainty
- Generated multiple outcome samples per input to identify the **Top-10 most probable scenarios**.
- Measured **uncertainty** via the diversity of sampled outcomes.

---

### 🔹 6. Autoencoder and Variational Autoencoder (VAE)
- Trained an **Autoencoder** to compress race features into a latent space.
- Extended to a **VAE**, learning a probabilistic latent space for generative modeling.
- VAE learns both means and variances for latent variables → allows random sampling.

---

### 🔹 7. Synthetic Data Generation
- Sampled from the VAE's latent space and decoded with the **generator network**.
- Produced **synthetic race data** that resemble real race patterns.
- These synthetic races can be reanalyzed or passed through the Bayesian Network to simulate realistic outcomes.
