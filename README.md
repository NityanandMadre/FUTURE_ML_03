# FUTURE_ML_03
# 🤖 Customer Support Chatbot

An AI-powered customer support chatbot built using NLP and Streamlit.  
The chatbot is trained on a real-world Twitter customer support dataset from Kaggle and can handle common customer queries such as refunds, account issues, and delivery problems.

---

## 🚀 Features
- NLP-based intent classification
- Real customer–agent conversation data
- Automatic response selection from dataset
- Cleaned and platform-neutral responses
- Interactive Streamlit web interface
- Modular and production-ready code structure

---

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- NLP (TF-IDF + Logistic Regression)

- 
---

## 📊 Dataset
**Twitter Customer Support Dataset – Kaggle**

- The dataset contains real customer queries and support agent responses.
- Original dataset is **not included** in this repository due to size constraints.
- Processed Q&A data is used for training and response generation.

---

## ▶️ How to Run the Project
### Install dependencies
```bash
pip install -r requirements.txt
python preprocess.py
streamlit run app.py
python test_model.py



