# EEG-based ADHD Classification using Deep Learning

This project presents a deep learning-based approach to classify Attention Deficit Hyperactivity Disorder (ADHD) and control subjects using EEG (Electroencephalogram) signals. The model combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) layers to learn both spatial and temporal features from EEG data.

---

## 🚀 Highlights

- ✅ Achieved **93.37% test accuracy** and **0.2073 test loss**.
- 📊 Balanced classification with **F1-score ≈ 0.93** for both classes.
- 🧠 Captures spatiotemporal patterns in EEG signals linked to ADHD.
- 🔬 Outperforms traditional ML baselines like SVM by a significant margin.

---

## 📂 Dataset

The EEG dataset used in this project contains recordings from both ADHD-diagnosed and control subjects. Each sample includes preprocessed EEG features extracted across time and channels.

- **Classes**: ADHD (1), Control (0)
- **Input Format**: Processed time-series EEG data
- **Train/Validation/Test Split**: Standard stratified split for evaluation

---

## 🧠 Model Architecture

The architecture is composed of:

- **CNN layers**: For spatial feature extraction across EEG channels
- **LSTM layers**: To capture temporal dependencies
- **Dense layers**: For final classification

```plaintext
Input → CNN → MaxPooling → Dropout → LSTM → Dense → Output
```

---

## 📊 Results

| Metric                 | SVM (Baseline)         | CNN+LSTM Model         |
|------------------------|------------------------|------------------------|
| Validation Accuracy     | 67.31%                 | **93.36%**             |
| Test Accuracy           | 68.00%                 | **93.37%**             |
| Validation F1-score     | 0.67                   | **0.93**               |
| Test F1-score           | 0.68                   | **0.93**               |
| Precision (Class 0/1)   | 0.71 / 0.63            | **0.92 / 0.93**        |
| Recall (Class 0/1)      | 0.70 / 0.63            | **0.95 / 0.90**        |

---

## 📈 Evaluation

- **Confusion Matrix**, **Classification Report**, and **Precision-Recall curves** are used to assess performance.
- Strong generalization on unseen data confirmed via held-out test set.

---

## 🛠️ Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/adhd-eeg-classification.git
cd adhd-eeg-classification
pip install -r requirements.txt
```

---

## 🧪 How to Run

Ensure your EEG dataset is placed in the appropriate directory. Then, to train the model:

```bash
python train.py
```

For evaluation on the test set:

```bash
python evaluate.py
```

---

## 📌 Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

---

## 📚 Folder Structure

```
├── data/                  # EEG dataset
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code (model, preprocessing)
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt
└── README.md
```

---

## 👨‍⚕️ Applications

- Clinical decision support for ADHD diagnosis  
- Neurological pattern recognition using EEG  
- Generalizable approach to EEG-based mental health diagnostics  

---

## 📄 License

This project is licensed under the MIT License.
