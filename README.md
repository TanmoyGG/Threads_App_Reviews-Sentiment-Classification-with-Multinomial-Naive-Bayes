<div align="center">

# 🧵 Threads App Reviews Sentiment Analysis

### *Decoding User Emotions with Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-red.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TanmoyGG/Threads_App_Reviews-Sentiment-Classification-with-Multinomial-Naive-Bayes/blob/main/Sentiment%20Classification(Naive%20Bayes).ipynb)

</div>

---


### Why This Project?

Understanding user sentiment is crucial for:
- **Product teams** to identify pain points and prioritize features
- **Marketing teams** to gauge public perception and brand reputation
- **Developers** to learn NLP techniques and sentiment classification workflows

### What Does It Do?

The project takes raw user reviews, processes them through multiple NLP preprocessing steps, trains a Multinomial Naive Bayes model with TF-IDF vectorization, and achieves **high accuracy** in predicting sentiment. It also provides visualizations of sentiment distribution, confusion matrices, and feature importance (most influential words).

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Framework** | Scikit-learn |
| **NLP Library** | NLTK (Natural Language Toolkit) |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |
| **Vectorization** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Algorithm** | Multinomial Naive Bayes |

---

## ✨ Features

- **📊 Comprehensive NLP Pipeline**: Complete text preprocessing including case folding, tokenization, punctuation removal, stopword filtering, stemming, and lemmatization
- **🎯 Balanced Classification**: Handles class imbalance using sample weights for improved model performance
- **📈 Advanced Feature Engineering**: TF-IDF vectorization with bigrams (1-2 grams) for capturing context
- **🔍 Feature Importance Analysis**: Visualizes the most influential words for positive and negative sentiments
- **📉 Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **🎨 Rich Visualizations**: 
  - Sentiment distribution bar charts
  - Confusion matrix heatmap
  - Feature importance plots
- **🧪 Interactive Predictions**: Test the model with custom text samples
- **⚖️ Smart Stopword Handling**: Retains negation words (not, no, nor, don't) crucial for sentiment analysis
- **🔄 Synonym Augmentation**: Optional data augmentation using WordNet synonyms
- **📐 Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score

---

## 🚀 Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package installer)
- **Jupyter Notebook or JupyterLab** (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TanmoyGG/Threads_App_Reviews-Sentiment-Classification-with-Multinomial-Naive-Bayes.git
   cd Threads_App_Reviews-Sentiment-Classification-with-Multinomial-Naive-Bayes
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn jupyter
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

### Usage

#### Running the Jupyter Notebook

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `Sentiment Classification(Naive Bayes).ipynb`
   - Click to open

3. **Update the data path** (if needed)
   - In the "Load dataset" section, update `DATA_PATH`:
   ```python
   DATA_PATH = 'threads_reviews.csv'  # Update if your file is in a different location
   ```

4. **Run all cells**
   - Click `Kernel` → `Restart & Run All`
   - Or run cells sequentially with `Shift + Enter`

#### Google Colab (No Installation Required!)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TanmoyGG/Threads_App_Reviews-Sentiment-Classification-with-Multinomial-Naive-Bayes/blob/main/Sentiment%20Classification(Naive%20Bayes).ipynb)

1. Click the badge above
2. Upload `threads_reviews.csv` to Colab
3. Update `DATA_PATH = '/content/threads_reviews.csv'`
4. Run all cells

#### Testing Custom Predictions

After training the model, you can test it with your own text:

```python
# Example usage
samples = [
    "I love how smooth the app feels now!",
    "Crashes every time I open it. Worst update ever.",
    "The interface is intuitive and easy to navigate"
]

predictions = predict_sentiment(samples)
print(predictions)  # Output: ['positive', 'negative', 'positive']
```

---

## 📁 Project Structure

```
Threads_App_Reviews-Sentiment-Classification-with-Multinomial-Naive-Bayes/
│
├── Sentiment Classification(Naive Bayes).ipynb    # Main Jupyter notebook
├── threads_reviews.csv                             # Dataset (34,099 reviews)
├── README.md                                       # Project documentation (this file)
└── requirements.txt                                # Python dependencies (optional)
```

### File Descriptions

| File | Description |
|------|-------------|
| **Sentiment Classification(Naive Bayes).ipynb** | Complete implementation including data loading, preprocessing, model training, evaluation, and visualization |
| **threads_reviews.csv** | Raw dataset containing Threads app reviews from Google Play Store with columns: `source`, `review_description`, `rating`, `review_date` |
| **README.md** | Comprehensive project documentation |

---

## 🔬 Methodology

### 1. **Data Loading & Preprocessing**
   - Load reviews from CSV with 34,099 entries
   - Map ratings to sentiments: 
     - Rating ≥ 4 → Positive
     - Rating ≤ 2 → Negative
     - Rating = 3 → Neutral (excluded)

### 2. **Text Preprocessing Pipeline**
   ```
   Raw Text → Case Folding → Tokenization → Punctuation Removal 
   → Stopword Removal → Lemmatization → TF-IDF Vectorization
   ```

### 3. **Feature Extraction**
   - **TF-IDF Vectorization** with unigrams and bigrams (1-2 grams)
   - Min document frequency: 2
   - Max document frequency: 90%

### 4. **Model Training**
   - **Algorithm**: Multinomial Naive Bayes
   - **Alpha**: 1.0 (Laplace smoothing)
   - **Class Balancing**: Computed sample weights to handle imbalanced data
   - **Train-Test Split**: 80-20 with stratification

### 5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - 5-Fold Stratified Cross-Validation
   - Feature Importance Analysis

---

## 📊 Results

The model achieves strong performance on the test set:

- **Accuracy**: ~88-92% (varies with preprocessing choices)
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across sentiments
- **F1-Score**: Strong harmonic mean indicating balanced performance

### Key Insights

**Top Positive Sentiment Words:**
- "love", "great", "good", "easy", "feature", "smooth", "best"

**Top Negative Sentiment Words:**
- "crash", "bad", "issue", "problem", "bug", "worst", "fix"

---

## 📚 Learning Outcomes

This project demonstrates:
- End-to-end NLP pipeline implementation
- Handling imbalanced datasets
- Feature engineering with TF-IDF
- Model evaluation and interpretation
- Real-world sentiment analysis application

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **Dataset Source**: Google Play Store Threads App Reviews
- **Libraries**: Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn
- **Inspiration**: Understanding user sentiment for product improvement

---

## 👨‍💻 Author

**TanmoyGG**

[![GitHub](https://img.shields.io/badge/GitHub-TanmoyGG-black?style=flat&logo=github)](https://github.com/TanmoyGG)

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and Python**

</div>