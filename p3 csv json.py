import json
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import gutenberg

     

# Handling Structured Data (CSV) - Using Seaborn's Titanic dataset
df_csv = sns.load_dataset("titanic")  # Load inbuilt Titanic dataset
print("\nStructured Data (CSV):\n", df_csv.head())

# Handling Semi-Structured Data (JSON) - Using Sample JSON Dictionary
data_json = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "Data Science", "Machine Learning"]
}
df_json = pd.json_normalize(data_json)  # Convert JSON to DataFrame
print("\nSemi-Structured Data (JSON):\n", df_json)

# Handling Unstructured Data (Text) - Using NLTK Gutenberg Corpus
nltk.download('gutenberg')
text_data = gutenberg.raw('shakespeare-hamlet.txt')  # Load Hamlet text
print("\nUnstructured Data (Text File):\n", text_data[:200])  # Show first 200 characters

     