import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
from joblib import load
from tkinter import PhotoImage

# Clean combined movies dataset
movies_clean_path = '/Users/samer/Documents/github_repos/Cinemalytics/data/clean/updated_clean_combined_movies.csv'
movies_clean_df = pd.read_csv(movies_clean_path)

# Clean combined series dataset
series_clean_path = '/Users/samer/Documents/github_repos/Cinemalytics/data/clean/clean_combined_series.csv'
series_clean_df = pd.read_csv(series_clean_path)

# Separate movie DataFrames by platform and content type
netflix_movies_df = movies_clean_df[movies_clean_df['platform'] == 'Netflix']
prime_video_movies_df = movies_clean_df[movies_clean_df['platform'] == 'Prime Video']
disney_plus_movies_df = movies_clean_df[movies_clean_df['platform'] == 'Disney+']

# Separate series DataFrames by platform and content type
netflix_series_df = series_clean_df[series_clean_df['platform'] == 'Netflix']
prime_video_series_df = series_clean_df[series_clean_df['platform'] == 'Prime Video']
disney_plus_series_df = series_clean_df[series_clean_df['platform'] == 'Disney+']











def load_data():
    dataset_path = '/Users/samer/Documents/github_repos/Cinemalytics/data/clean/updated_clean_combined_movies.csv'
    return pd.read_csv(dataset_path)

def process_data(df):
    df['genre'] = df['genre'].astype(str).str.split(',')
    exploded_genres = df.explode('genre')
    exploded_genres['genre'] = exploded_genres['genre'].str.strip()
    unique_genres = sorted(exploded_genres['genre'][exploded_genres['genre'] != 'nan'].unique())
    unique_age_ratings = sorted(df['age_rating'].dropna().unique())
    return unique_genres, unique_age_ratings

def load_image(root):
    global logo_image
    logo_path = '/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png'
    # Open the original image
    original_logo = Image.open(logo_path)
    # Resize the image using Image.Resampling.LANCZOS
    resized_logo = original_logo.resize((100, 100), Image.Resampling.LANCZOS)
    # Convert the resized image to a PhotoImage
    logo_image = ImageTk.PhotoImage(resized_logo, master=root)

def setup_gui(unique_genres, unique_age_ratings, clf, label_encoders):
    root = tk.Tk()
    root.title("Cinemalytics")

    load_image(root)  # Load the logo image after the root window is created

    logo_label = tk.Label(root, image=logo_image)
    logo_label.grid(row=0, column=0, columnspan=2, pady=10)

    create_widgets(root, unique_genres, unique_age_ratings, clf, label_encoders)

    root.mainloop()

def create_widgets(root, unique_genres, unique_age_ratings, clf, label_encoders):
    genre_var, duration_entry, age_rating_var, result_label = create_input_widgets(root, unique_genres, unique_age_ratings)
    setup_predict_button(root, clf, genre_var, duration_entry, age_rating_var, label_encoders, result_label)

def create_input_widgets(root, unique_genres, unique_age_ratings):
    tk.Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=10)
    genre_var = tk.StringVar()
    ttk.Combobox(root, textvariable=genre_var, values=unique_genres).grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Age Rating:").grid(row=2, column=0, padx=10, pady=10)
    age_rating_var = tk.StringVar()
    ttk.Combobox(root, textvariable=age_rating_var, values=unique_age_ratings).grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Duration (min):").grid(row=3, column=0, padx=10, pady=10)
    duration_entry = tk.Entry(root)
    duration_entry.grid(row=3, column=1, padx=10, pady=10)

    result_label = tk.Label(root, text="Prediction will appear here")
    result_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    return genre_var, duration_entry, age_rating_var, result_label


def setup_predict_button(root, clf, genre_var, duration_entry, age_rating_var, label_encoders, result_label):
    predict_button = tk.Button(root, text="Predict Platform", command=lambda: predict_platform(clf, genre_var.get(), duration_entry.get(), age_rating_var.get(), label_encoders, result_label))
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

def predict_platform(clf, genre_input, duration_input, age_rating_input, label_encoders, result_label):
    label_encoder_genre, label_encoder_age_rating, label_encoder_platform = label_encoders
    # Check for missing inputs first
    if not genre_input or not duration_input or not age_rating_input:
        result_label.config(text="Please ensure to enter something for all information fields.")
        return
    try:
        duration_input = int(duration_input)
        genre_encoded = label_encoder_genre.transform([genre_input])[0]
        age_rating_encoded = label_encoder_age_rating.transform([age_rating_input])[0]
        input_features = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'duration_min': [duration_input],
            'age_rating_encoded': [age_rating_encoded]
        })
        predicted_platform_encoded = clf.predict(input_features)
        predicted_platform = label_encoder_platform.inverse_transform(predicted_platform_encoded)
        result_label.config(text=f"Best Platform for the new movie: {predicted_platform[0]}")
    except ValueError as e:
        result_label.config(text=f"Error: {e}. Please ensure your inputs match the dataset's categories and that duration is a number.")
    except Exception as e:
        result_label.config(text="There has been an error. Please try again later. If the issue persists, please contact the developers.")


if __name__ == "__main__":
    df = load_data()
    unique_genres, unique_age_ratings = process_data(df)
    clf = load(r'/Users/samer/Documents/model_cinemalytics/random_forest_model.joblib')
    label_encoder_genre = load(r'/Users/samer/Documents/model_cinemalytics/genre_encoder.joblib')
    label_encoder_age_rating = load(r'/Users/samer/Documents/model_cinemalytics/age_rating_encoder.joblib')
    label_encoder_platform = load(r'/Users/samer/Documents/model_cinemalytics/platform_encoder.joblib')
    label_encoders = (label_encoder_genre, label_encoder_age_rating, label_encoder_platform)
    
    # Now setup_gui is responsible for initializing the Tkinter root and loading the image
    setup_gui(unique_genres, unique_age_ratings, clf, label_encoders)
