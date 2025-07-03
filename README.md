# Fake-News-Detection-using-DeepLearning
Fake News Detection Using Deep Learning
This repository contains the implementation and research of the paper titled "Fake News Detection Using Deep Learning", which proposes a novel method using an Attention-Based Convolutional Bidirectional Long Short-Term Memory (AC-BiLSTM) model to identify fake news from real news articles.

📌 Abstract
With the rise of social media, the spread of misinformation and fake news has become a serious issue. This project presents a deep learning approach using AC-BiLSTM to detect whether a given news article is fake or real. The model processes news text in both forward and backward directions, enhancing contextual understanding.

🧠 Model Architecture
Embedding Layer: GloVe Embeddings (100d)

Bidirectional LSTM (BiLSTM): Captures forward and backward semantics

Attention Layer: Highlights significant words contributing to classification

Dropout & Dense Layers: Prevent overfitting and produce final classification

🧪 Dataset
Dataset used: fake_and_real_news.xlsx

Classes: Real (0), Fake (1), Neutral

Source: Collected and preprocessed from real-world news platforms and fake news datasets.

🛠️ Implementation Steps
python
Copy
Edit
1. Import Libraries (torch, numpy, pandas, sklearn, etc.)
2. Load and preprocess the dataset (tokenization, padding, splitting)
3. Define a custom dataset class
4. Build the BiLSTM model using PyTorch/Keras
5. Train using Adam Optimizer & CrossEntropyLoss
6. Evaluate model using metrics like F1 Score, Accuracy, Precision-Recall
📊 Results
Training Accuracy: ~85%

F1 Score:

Fake News: 0.45

Real News: 0.45

Neutral: 0.25

Precision-Recall Curve: Indicates balanced learning

Loss Graphs: Show convergence with minimal overfitting

📈 Visualizations
📉 Loss Graphs (Training vs Validation)

🎯 Accuracy Graphs

🧪 Precision-Recall Curves

🧾 F1 Scores by Category

📚 References
The methodology is built on prior work by researchers such as Shu et al. (2017–2020), Ruchansky et al. (2017), and Vosoughi et al. (2018). For a complete list of references, see the References section in the paper.

✍️ Authors: 
Md. Mubashera

Shaik Karishma

Shaik Tabassum

M Pravallika

Y. Swathi (Guide)
Vignan's Nirula Institute of Technology and Science for Women
