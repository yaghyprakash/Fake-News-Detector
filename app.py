import io
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st._config.set_option("server.maxUploadSize", 1000)  # MB

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.write("A machine-learning demonstration that classifies news text using TF-IDF and Logistic Regression.")

st.warning(
    "This app identifies patterns learned from its training dataset. It does not verify facts. "
    "A 'real' prediction does not prove that a story is true, and a 'fake' prediction does not prove it is false."
)

@st.cache_resource(show_spinner="Training the model...")
def train_model(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Recognize common column names.
    cols = {str(c).strip().lower(): c for c in df.columns}
    text_col = next((cols[k] for k in ("text", "article", "content", "news", "title") if k in cols), None)
    label_col = next((cols[k] for k in ("label", "class", "target", "type") if k in cols), None)

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not identify the text and label columns. "
            "The CSV needs columns such as 'text' and 'label'."
        )

    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data[label_col] = data[label_col].astype(str).str.strip().str.lower()

    label_map = {
        "fake": "fake", "false": "fake", "0": "fake", "fake news": "fake", "unreliable": "fake",
        "real": "real", "true": "real", "1": "real", "true news": "real", "reliable": "real"
    }
    data["label"] = data[label_col].map(label_map)
    data = data.dropna(subset=["label"])
    data = data[data[text_col].str.len() >= 10]
    data = data.drop_duplicates(subset=[text_col, "label"])

    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both fake and real labels.")
    if data["label"].value_counts().min() < 2 or len(data) < 10:
        raise ValueError("The dataset is too small. Provide at least 10 rows and at least 2 examples per class.")

    X_train, X_test, y_train, y_test = train_test_split(
        data[text_col], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 1), min_df=3, sublinear_tf=True)),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, y_test, predictions, len(data), data["label"].value_counts()

with st.sidebar:
    st.header("1. Upload training data")
    st.write("Upload `news_dataset.csv`, created by running `combine.py`.")
    uploaded = st.file_uploader("Choose a CSV dataset", type=["csv"])
    st.caption("Required columns: text, label. Labels: fake and real.")

if uploaded is None:
    st.info("Upload your labelled CSV dataset using the sidebar to train the model.")
    st.subheader("Expected CSV structure")
    st.dataframe(pd.DataFrame({
        "text": [
            "The city council approved the annual public transport budget after a recorded vote.",
            "A secret potion guarantees that everyone can live forever, scientists confirm."
        ],
        "label": ["real", "fake"]
    }), use_container_width=True)
    st.caption("These two rows only demonstrate the format. They are not enough to train a useful model.")
    st.stop()

try:
    model, accuracy, y_test, predictions, row_count, counts = train_model(uploaded.getvalue())

    st.success(f"Model trained using {row_count:,} labelled articles.")
    left, right = st.columns(2)
    left.metric("Test-set accuracy", f"{accuracy * 100:.2f}%")
    right.metric("Articles used", f"{row_count:,}")

    with st.expander("Training data distribution"):
        st.write(counts.rename(index={"fake": "Fake", "real": "Real"}))
    with st.expander("Evaluation details"):
        st.text(classification_report(y_test, predictions, labels=["fake", "real"], zero_division=0))
        matrix = confusion_matrix(y_test, predictions, labels=["fake", "real"])
        st.write("Confusion matrix (rows = actual; columns = predicted)")
        st.dataframe(pd.DataFrame(
            matrix,
            index=["Actual fake", "Actual real"],
            columns=["Predicted fake", "Predicted real"]
        ), use_container_width=True)

    st.divider()
    st.header("2. Analyze a news article")
    article = st.text_area(
        "Paste a headline or article text",
        height=220,
        placeholder="Paste the news article here..."
    )

    if st.button("Analyze article", type="primary"):
        if len(article.strip()) < 30:
            st.error("Please enter at least 30 characters.")
        else:
            prediction = model.predict([article])[0]
            probability_values = model.predict_proba([article])[0]
            class_names = list(model.named_steps["classifier"].classes_)
            confidence = float(probability_values[class_names.index(prediction)]) * 100

            if prediction == "fake":
                st.error(f"Prediction: FAKE-style text — model confidence {confidence:.1f}%")
            else:
                st.success(f"Prediction: REAL-style text — model confidence {confidence:.1f}%")

            st.progress(confidence / 100)
            st.caption(
                "Model confidence is not the probability that the claim is true or false. "
                "Check important claims against independent, reliable sources."
            )

except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.info(
        "Check that your CSV has article text and labels, both classes are present, "
        "and the file is a valid CSV."
    )
