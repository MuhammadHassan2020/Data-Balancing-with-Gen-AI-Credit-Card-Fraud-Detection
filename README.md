#  Data Balancing with Generative AI — Credit Card Fraud Detection

A hands-on project that demonstrates how **Generative Adversarial Networks (GANs)** can be used to synthetically balance highly imbalanced credit card transaction data, particularly to detect **fraudulent transactions**. This project covers the complete data pipeline from preprocessing to GAN training and synthetic data analysis.

---

## 📌 Business Problem

Fraudulent transactions are rare but critical events, making fraud detection a classic **class imbalance problem** in machine learning. Traditional models often underperform due to limited examples of fraudulent behavior. The goal is to use **Generative AI** to **generate synthetic fraud samples**, thus improving classifier performance by providing more balanced training data.

---

## 🧠 Solution Summary

- Used **GANs** to generate high-quality synthetic fraud transactions.
- Balanced dataset improved fraud detection capability without overfitting.
- Compared **real vs synthetic distributions** using PCA and visualization techniques.
- Verified feature space similarity to ensure generated samples are meaningful.
- Demonstrated the **impact of synthetic data** on real-world fraud classification tasks.

---

## 🔍 Key Tasks Performed

- ✅ Data Cleaning: Removed nulls, dropped irrelevant columns (e.g., `Time`), scaled `Amount`.
- ✅ Data Exploration: Visualized class distribution & applied PCA for fraud pattern inspection.
- ✅ GAN Modeling:
  - Designed a custom **Generator** to produce synthetic fraud features.
  - Built a **Discriminator** to distinguish between real and fake fraud.
  - Combined both in a GAN architecture for adversarial training.
- ✅ Evaluation:
  - Trained the GAN and tracked convergence via PCA scatter plots.
  - Generated 1000 synthetic fraud samples for comparison.
  - Visualized real vs synthetic feature distributions.

---

## 🛠️ Technologies & Libraries

- Python, NumPy, Pandas, Matplotlib, Seaborn  
- TensorFlow / Keras (for GAN implementation)  
- Principal Component Analysis (PCA)  
- GAN architecture (Generator + Discriminator)

---

## 📸 Sample Output Visualization

*(Insert your GAN training visual, PCA plots, or feature distribution images here)*  
For example:  
```markdown
![Fraud Data PCA](images/fraud_pca.png)
