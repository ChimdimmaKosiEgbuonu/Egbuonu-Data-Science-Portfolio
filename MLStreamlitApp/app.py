# I used Generative AI in assistance with the coding portion of the portfolio, but all written parts (such as the ReadMe) are my own work and did not use Generative AI.
# imports all of the needed libraries for this portfolio update
import pandas as pd           
import seaborn as sns          
import matplotlib.pyplot as plt  
import streamlit as st 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Lets the user use pre-loaded dataset or upload one of their choosing 
st.sidebar.header("Choosing the Data")
selection = st.sidebar.radio("Choose a data source:", ["Sample Dataset (Tips)", "Upload CSV"])

# lets user use "tips" dataset or upload their own
if selection == "Sample Dataset (Tips)":
    tips_openml = fetch_openml(data_id=42464, as_frame=True, parser="auto")
    df = tips_openml.frame
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file in the sidebar or select the Sample Dataset.")
        st.stop()

# provides preview of the dataset
st.subheader("Preview of Data")
st.dataframe(df.head())

# 2. selects target column for predictions
st.sidebar.header("Targeting and Training!")
default_index = list(df.columns).index("tip") if "tip" in df.columns else len(df.columns) - 1
target_col = st.sidebar.selectbox("Select your target column (y):", df.columns, index=default_index)

test_size = st.sidebar.slider("Test set size", 10, 50, 20, step=5) / 100

# decision tree hyperparameters
st.sidebar.header("Model Hyperparameters")
max_depth = st.sidebar.slider("Maximum Depth", min_value=1, max_value=15, value=3)
min_samples_split = st.sidebar.slider("Minimum Samples Split", min_value=2, max_value=20, value=2)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

#cleans the data and prepares  X and y
df_clean = df.dropna().copy()

# splits features (X) and target (y)
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col].astype(str)


X = pd.get_dummies(X, drop_first=True)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# decision tree model
model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    criterion=criterion,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

st.subheader("Performance of the Model on Test Data")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")

# explanations of the metrics
st.markdown(f"""
**Description of Metrics:**
** Accuracy:** The proportion of total predictions a model gets right out of all predictions
**Precision:** A metric that measures the accuracy of positive predictions
**Recall:** The rate in which a model is able to find the true positive cases within a dataset
""")

# confusion Matrix and decision tree
st.subheader("Confusion Matrix")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

st.subheader("Decision Tree Diagram")
fig2, ax2 = plt.subplots(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=labels,
    filled=True,
    fontsize=8,
    ax=ax2
)
st.pyplot(fig2)
