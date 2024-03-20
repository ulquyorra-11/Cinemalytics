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

def load_image(path, root, size=(100, 100)):
    original_logo = Image.open(path)
    resized_logo = original_logo.resize(size, Image.Resampling.LANCZOS)
    logo_image = ImageTk.PhotoImage(resized_logo, master=root)
    return logo_image


def setup_gui(unique_genres, unique_age_ratings, clf, clf_regressor, label_encoders):
    root = tk.Tk()
    root.title("Cinemalytics")

    # Load the logo image here, using the corrected function
    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', root, size=(300, 300))
    
    logo_label = tk.Label(root, image=logo_image)
    logo_label.grid(row=0, column=0, columnspan=2, pady=10)
    # Make sure the image reference is kept to prevent garbage collection
    logo_label.image = logo_image

    create_widgets(root, unique_genres, unique_age_ratings, clf, clf_regressor, label_encoders)
    
    root.mainloop()


def create_widgets(root, unique_genres, unique_age_ratings, clf, clf_regressor, label_encoders):
    # Inputs
    tk.Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=10)
    genre_var = tk.StringVar()
    ttk.Combobox(root, textvariable=genre_var, values=unique_genres).grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Duration (min):").grid(row=2, column=0, padx=10, pady=10)
    duration_entry = tk.Entry(root)
    duration_entry.grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Age Rating:").grid(row=3, column=0, padx=10, pady=10)
    age_rating_var = tk.StringVar()
    ttk.Combobox(root, textvariable=age_rating_var, values=unique_age_ratings).grid(row=3, column=1, padx=10, pady=10)

    # Outputs
    result_label = tk.Label(root, text="Prediction will appear here")
    result_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    result_label_2 = tk.Label(root, text="Revenue prediction will appear here")
    result_label_2.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    # Predict Button
    predict_button = tk.Button(root, text="Predict Platform", command=lambda: predict_platform(clf, clf_regressor, genre_var.get(), duration_entry.get(), age_rating_var.get(), label_encoders, result_label, result_label_2))
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

def predict_platform(clf, clf_regressor, genre_input, duration_input, age_rating_input, label_encoders, result_label, result_label_2):
    label_encoder_genre, label_encoder_age_rating, label_encoder_platform = label_encoders
    try:
        # Input validation and encoding
        duration_input = int(duration_input)
        genre_encoded = label_encoder_genre.transform([genre_input])[0]
        age_rating_encoded = label_encoder_age_rating.transform([age_rating_input])[0]

        # Predict platform
        input_features = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'duration_min': [duration_input],
            'age_rating_encoded': [age_rating_encoded]
        })
        predicted_platform_encoded = clf.predict(input_features)
        predicted_platform = label_encoder_platform.inverse_transform(predicted_platform_encoded)
        result_label.config(text=f"Best Platform for the movie: {predicted_platform[0]}")

        # Predict revenue
        platform_encoded = predicted_platform_encoded
        input_features_2 = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'duration_min': [duration_input],
            'age_rating_encoded': [age_rating_encoded],
            'platform_encoded' : [platform_encoded]
        })
        predicted_revenue = clf_regressor.predict(input_features_2)
        result_label_2.config(text=f"Estimated revenue: ${predicted_revenue[0]:,.2f}")
        
    except ValueError as e:
        result_label.config(text=f"Error: {e}. Please ensure all inputs are correctly formatted.")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

if __name__ == "__main__":
    df = load_data()
    unique_genres, unique_age_ratings = process_data(df)
    clf = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/random_forest_model.joblib')
    clf_regressor = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/random_forest_regressor_with_revenue.joblib')
    label_encoder_genre = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/genre_encoder.joblib')
    label_encoder_age_rating = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/age_rating_encoder.joblib')
    label_encoder_platform = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/platform_encoder.joblib')
    label_encoders = (label_encoder_genre, label_encoder_age_rating, label_encoder_platform)
    
    setup_gui(unique_genres, unique_age_ratings, clf, clf_regressor, label_encoders)
