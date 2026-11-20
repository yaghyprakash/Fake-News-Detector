# Fake News Detector

A machine-learning web application built with **Python**, **Streamlit**,
and **scikit-learn** that classifies news text as **Fake** or **Real**.

The project uses Natural Language Processing (NLP) techniques to convert
news text into numerical features and a Logistic Regression model to
make predictions.

## Features

-   Classifies news text as Fake or Real.
-   Provides a simple web interface built with Streamlit.
-   Uses TF-IDF to convert text into numerical features.
-   Uses Logistic Regression for classification.
-   Splits data into training and testing sets.
-   Evaluates the model using metrics such as accuracy, precision,
    recall, F1-score, and a confusion matrix (where implemented).

## Technologies Used

  Technology            Purpose
  --------------------- ---------------------------------------
  Python                Main programming language
  Streamlit             Web application interface
  Pandas                Data loading and processing
  Scikit-learn          Machine-learning model and evaluation
  TF-IDF                Text feature extraction
  Logistic Regression   News classification

## Project Structure

``` text
fake-news-detector/
│
├── app.py              # Main Streamlit application
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (if included)
```

If your project uses additional Python files or datasets, add them to
this structure.

## Installation

### 1. Install Python

Install Python 3.10 or later from the official Python website:

https://www.python.org/downloads/

During installation on Windows, enable **Add Python to PATH**.

### 2. Download or clone the repository

``` bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd fake-news-detector
```

Alternatively, download the repository as a ZIP file and extract it.

### 3. Install dependencies

Open a terminal in the project folder and run:

``` bash
python -m pip install streamlit pandas scikit-learn
```

If a `requirements.txt` file is included, you can instead run:

``` bash
python -m pip install -r requirements.txt
```

## Running the Application

From the project directory, run:

``` bash
python -m streamlit run app.py
```

Streamlit will start a local server and display a URL, usually:

``` text
http://localhost:8501
```

Open that address in your browser to use the application.

## How It Works

1.  **Data input:** The application receives news text and, depending on
    the app configuration, may load a labelled news dataset.
2.  **Text preprocessing and feature extraction:** TF-IDF (Term
    Frequency--Inverse Document Frequency) converts text into numerical
    features that represent the importance of words.
3.  **Model training:** Logistic Regression learns patterns from the
    labelled training data.
4.  **Prediction:** The trained model processes the supplied news text
    and predicts whether it is Fake or Real.
5.  **Evaluation:** The model can be evaluated on test data using
    classification metrics.

## Machine-Learning Approach

### TF-IDF Vectorization

Machine-learning algorithms work with numerical data rather than raw
sentences. TF-IDF converts text into numerical feature vectors. It gives
higher importance to terms that are useful for distinguishing documents
and reduces the influence of words that occur frequently across many
documents.

### Logistic Regression

Logistic Regression is a supervised classification algorithm. In this
project, it learns from labelled examples of fake and real news and uses
the extracted text features to classify new input.

## Dataset

The model requires a labelled dataset containing news text and its
corresponding class labels.

A typical dataset may contain columns such as:

-   `text` --- the news article or headline.
-   `label` --- the class assigned to the article (for example, Fake or
    Real).

**Important:** The exact column names and label format must match what
the application expects. Check `app.py` before using a different
dataset.

## Model Performance

The model's accuracy and other evaluation scores depend on the dataset,
preprocessing, train-test split, and model configuration.

No fixed accuracy is claimed in this README. Run the application or
evaluation code on your dataset and report the measured results here.

  Metric      Result
  ----------- ---------------------
  Accuracy    Add measured result
  Precision   Add measured result
  Recall      Add measured result
  F1-score    Add measured result

## Limitations

-   The model learns patterns from its training dataset; it does not
    independently verify claims against reliable sources.
-   It may misclassify satire, opinions, breaking news, or articles on
    topics that are underrepresented in the dataset.
-   Dataset bias, duplicate articles, and incorrect labels can affect
    performance.
-   A prediction is not proof that an article is true or false.

**This application is an educational machine-learning project, not a
professional fact-checking service.**

## Future Improvements

-   Improve text preprocessing and dataset quality.
-   Test additional classification algorithms.
-   Add cross-validation and more detailed model evaluation.
-   Detect duplicate or near-duplicate articles.
-   Add explainability to show which text features influenced a
    prediction.
-   Deploy the application so it can be accessed online.

## Contributing

Contributions and suggestions are welcome. To contribute:

1.  Fork the repository.
2.  Create a new branch.
3.  Make your changes.
4.  Submit a pull request describing the changes.

## License

Add a license before distributing or reusing this project. If no license
is included, do not assume that the code is available for unrestricted
reuse.

------------------------------------------------------------------------

**Developed as a Python and Machine Learning project.**
