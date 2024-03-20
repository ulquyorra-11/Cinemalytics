import pandas as pd
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Frame
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
def center_window(root, width=1200, height=600):
    # Get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width/2 - width/2)
    center_y = int(screen_height/2 - height/2)

    # Set the position of the window to the center of the screen
    root.geometry(f'{width}x{height}+{center_x}+{center_y}')

# Function to create the start window
def start_window():
    root = Tk()
    root.attributes("-alpha", 0.9)  # Set window transparency
    center_window(root, width=1200, height=600)  # Set window size and center it
    root.title("Cinemalytics")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(300, 300))
    logo_label = Label(root, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=50)

    start_button = Button(root, text="Start", command=lambda: platform_window(root))
    start_button.pack(pady=20)

    root.mainloop()

# Function to create the Platform Prediction window
def platform_window(previous_root):
    previous_root.destroy()
    root = Tk()
    root.attributes("-alpha", 0.9)  # Set window transparency
    center_window(root)  # Set window size and center it
    root.title("Platform Prediction")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(300, 300))
    logo_label = Label(root, image=logo_image)
    logo_label.image = logo_image
    logo_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

    # Store user inputs
    genre_var = StringVar()
    age_rating_var = StringVar()
    duration_var = StringVar()

    # Create widgets
    Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=5)  # Adjusted pady
    genre_combobox = Combobox(root, textvariable=genre_var, values=unique_genres)
    genre_combobox.grid(row=1, column=1, padx=10, pady=5)  # Adjusted pady

    Label(root, text="Age Rating:").grid(row=2, column=0, padx=10, pady=5)  # Adjusted pady
    age_rating_combobox = Combobox(root, textvariable=age_rating_var, values=unique_age_ratings)
    age_rating_combobox.grid(row=2, column=1, padx=10, pady=5)  # Adjusted pady

    Label(root, text="Duration (min):").grid(row=3, column=0, padx=10, pady=5)  # Adjusted pady
    duration_entry = Entry(root, textvariable=duration_var)
    duration_entry.grid(row=3, column=1, padx=10, pady=5)  # Adjusted pady

    predict_button = Button(root, text="Predict Platform", command=lambda: validate_and_predict(root, genre_var.get(), duration_var.get(), age_rating_var.get()))
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

    # Center all elements vertically and horizontally
    for i in range(5):  # Adjusted range to match number of rows
        root.grid_rowconfigure(i, weight=1)
    for j in range(2):  # Adjusted range to match number of columns
        root.grid_columnconfigure(j, weight=1)

    root.mainloop()

# Function to validate the inputs and predict the platform
def validate_and_predict(root, genre_input, duration_input, age_rating_input):
    if not genre_input or not duration_input or not age_rating_input:
        messagebox.showerror("Input Error", "Please enter all the required fields.")
        return

    try:
        int(duration_input)  # Just to check if it's an integer
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a full number for duration.")
        return

    predict_platform(root, genre_input, duration_input, age_rating_input)

# Function to predict the platform and display the result
def predict_platform(previous_root, genre_input, duration_input, age_rating_input):
    if not genre_input or not duration_input or not age_rating_input:
        messagebox.showerror("Input Error", "All fields are required.")
        return

    try:
        duration = int(duration_input)  # Check if duration input is an integer
    except ValueError:
        messagebox.showerror("Input Error", "Duration must be a number.")
        return

    try:
        genre_encoded = label_encoder_genre.transform([genre_input])[0]
        age_rating_encoded = label_encoder_age_rating.transform([age_rating_input])[0]
        input_features = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'duration_min': [duration],
            'age_rating_encoded': [age_rating_encoded]
        })
        predicted_platform_encoded = clf.predict(input_features)
        predicted_platform = label_encoder_platform.inverse_transform(predicted_platform_encoded)[0]

        # After getting the prediction, display the result
        previous_root.destroy()
        result_window(predicted_platform)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Please ensure all inputs are correctly formatted.\n{e}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction.\n{e}")

def result_window(platform):
    root = Tk()
    root.attributes("-alpha", 0.9)  # Set window transparency
    center_window(root, width=1200, height=600)  # Set window size and center it
    root.title(f"{platform} Prediction")

    # Load App Logo
    app_logo_path = '/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png'
    app_logo_image = load_image(app_logo_path, size=(300, 300))
    app_logo_label = Label(root, image=app_logo_image)
    app_logo_label.image = app_logo_image
    app_logo_label.pack(pady=50)

    platform_logos = {
        'Netflix': '/Users/samer/Documents/github_repos/Cinemalytics/images/netflix_logo_4.png',
        'Prime Video': '/Users/samer/Documents/github_repos/Cinemalytics/images/prime_video_logo1.webp',
        'Disney+': '/Users/samer/Documents/github_repos/Cinemalytics/images/disney_plus_logo1.png'
    }

    logo_path = platform_logos.get(platform)
    if logo_path:
        logo_image = load_image(logo_path)
        logo_label = Label(root, image=logo_image)
        logo_label.image = logo_image
        logo_label.pack(pady=20)

    message_label = Label(root, text=f"Based on your criteria, the best platform for your movie is {platform}")
    message_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_window()
