import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
preprocessed_dataset_folder = "../Data/PreprocessedDataset"
movies_df = pd.read_csv(preprocessed_dataset_folder+"/movies.csv")
import streamlit


# Function to preprocess plot text
def preprocess_plot_text(plot_text):
    tokens = word_tokenize(plot_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# May be useful for individual recommendations
def visualize_wordcloud_for_movie_set(movie_indices, movies_df):
    combined_plot_text = ""

    for index in movie_indices:
        movie_row = movies_df.iloc[index]

        cleaned_plot = preprocess_plot_text(movie_row['plot'])
        combined_plot_text += cleaned_plot + " "

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_plot_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for Plot")
    plt.show()


