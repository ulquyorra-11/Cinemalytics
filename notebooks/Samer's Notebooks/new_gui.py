import pandas as pd
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox
from PIL import Image, ImageTk
from joblib import load
from tkinter.ttk import Combobox

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

def load_image(path, size=(100, 100)):
    original_logo = Image.open(path)
    resized_logo = original_logo.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized_logo)

# Global variables
df = load_data()
unique_genres, unique_age_ratings = process_data(df)
clf = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/random_forest_model.joblib')
clf_regressor = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/random_forest_regressor_with_revenue.joblib')
label_encoder_genre = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/genre_encoder.joblib')
label_encoder_age_rating = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/age_rating_encoder.joblib')
label_encoder_platform = load('/Users/samer/Documents/github_repos/Cinemalytics/trained_models/platform_encoder.joblib')
label_encoders = (label_encoder_genre, label_encoder_age_rating, label_encoder_platform)

# Function to center the window
def center_window(root, width=1200, height=650):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - width/2)
    center_y = int(screen_height/2 - height/2)
    root.geometry(f'{width}x{height}+{center_x}+{center_y}')

def start_window():
    root = Tk()
    root.attributes("-alpha", 0.9)
    center_window(root, width=1200, height=650)
    root.title("Cinemalytics")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(250, 250))
    logo_label = Label(root, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=50)

    start_button = Button(root, text="Start", command=lambda: platform_window(root))
    start_button.pack(pady=20)

    root.mainloop()

def platform_window(previous_root):
    previous_root.destroy()
    root = Tk()
    root.attributes("-alpha", 0.9)
    center_window(root)
    root.title("Platform Prediction")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(250, 250))
    logo_label = Label(root, image=logo_image)
    logo_label.image = logo_image
    logo_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

    genre_var = StringVar()
    age_rating_var = StringVar()
    duration_var = StringVar()

    Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=5)
    genre_combobox = Combobox(root, textvariable=genre_var, values=unique_genres)
    genre_combobox.grid(row=1, column=1, padx=10, pady=5)

    Label(root, text="Age Rating:").grid(row=2, column=0, padx=10, pady=5)
    age_rating_combobox = Combobox(root, textvariable=age_rating_var, values=unique_age_ratings)
    age_rating_combobox.grid(row=2, column=1, padx=10, pady=5)

    Label(root, text="Duration (min):").grid(row=3, column=0, padx=10, pady=5)
    duration_entry = Entry(root, textvariable=duration_var)
    duration_entry.grid(row=3, column=1, padx=10, pady=5)
    predict_button = Button(root, text="Predict Platform", command=lambda: validate_and_predict(root, genre_var.get(), duration_var.get(), age_rating_var.get()))
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

    for i in range(5):
        root.grid_rowconfigure(i, weight=1)
    for j in range(2):
        root.grid_columnconfigure(j, weight=1)

    root.mainloop()

def validate_and_predict(root, genre_input, duration_input, age_rating_input):
    if not genre_input or not duration_input or not age_rating_input:
        messagebox.showerror("Input Error", "Please enter all the required fields.")
        return

    try:
        duration = int(duration_input)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a full number for duration.")
        return

    predict_platform(root, genre_input, duration_input, age_rating_input)

def predict_platform(previous_root, genre_input, duration_input, age_rating_input):
    if not genre_input or not duration_input or not age_rating_input:
        messagebox.showerror("Input Error", "All fields are required.")
        return

    try:
        duration = int(duration_input)
        genre_encoded = label_encoder_genre.transform([genre_input])[0]
        age_rating_encoded = label_encoder_age_rating.transform([age_rating_input])[0]
        input_features = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'duration_min': [duration],
            'age_rating_encoded': [age_rating_encoded]
        })
        predicted_platform_encoded = clf.predict(input_features)
        predicted_platform = label_encoder_platform.inverse_transform(predicted_platform_encoded)[0]

        input_features['platform_encoded'] = predicted_platform_encoded
        predicted_revenue = clf_regressor.predict(input_features)[0]

        previous_root.destroy()
        result_window(predicted_platform, predicted_revenue, genre_input, age_rating_input, duration_input)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Please ensure all inputs are correctly formatted.\n{e}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction.\n{e}")

def result_window(platform, predicted_revenue, genre, age_rating, duration):
    root = Tk()
    root.attributes("-alpha", 0.9)
    center_window(root, width=1200, height=650)
    root.title(f"{platform} Prediction")

    app_logo_path = '/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png'
    app_logo_image = load_image(app_logo_path, size=(250, 250))
    app_logo_label = Label(root, image=app_logo_image)
    app_logo_label.image = app_logo_image
    app_logo_label.pack(pady=50)

    platform_logos = {
        'Netflix': '/Users/samer/Documents/github_repos/Cinemalytics/images/netflix_logo_4.png',
        'Prime Video': '/Users/samer/Documents/github_repos/Cinemalytics/images/prime_video_logo1.webp',
        'Disney+': '/Users/samer/Documents/github_repos/Cinemalytics/images/disney_plus_logo1.png',
    }

    logo_path = platform_logos.get(platform)
    if logo_path:
        logo_image = load_image(logo_path)
        logo_label = Label(root, image=logo_image)
        logo_label.image = logo_image
        logo_label.pack(pady=20)

    message_label = Label(root, text=f"Based on your movie with the genre {genre}, age rating {age_rating}, and duration {duration} minutes,\nthe best platform for your movie is {platform}. The estimated revenue for your movie in this region is: ${predicted_revenue:,.2f}")
    message_label.pack(pady=10)

    # Add a "Back to Start" button
    back_button = Button(root, text="Back to Start", command=lambda: restart_app(root))
    back_button.pack(pady=20)

    root.mainloop()

def restart_app(current_root):
    current_root.destroy()
    start_window()


if __name__ == "__main__":
    start_window()
