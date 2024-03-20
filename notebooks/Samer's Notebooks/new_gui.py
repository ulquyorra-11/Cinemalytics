import pandas as pd
from tkinter import Tk, Label, Button, Entry, Toplevel, StringVar
from PIL import Image, ImageTk
from joblib import load
from tkinter.ttk import Combobox
from tkinter import messagebox

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
    center_window(root)  # This line centers the window and sets the size
    root.title("Cinemalytics")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(200, 200))
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
    root.geometry('600x400')  # Set the window size
    root.title("Platform Prediction")

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(200, 200))
    logo_label = Label(root, image=logo_image)
    logo_label.image = logo_image
    logo_label.grid(row=0, column=0, columnspan=2)

    # Store user inputs
    genre_var = StringVar()
    age_rating_var = StringVar()
    duration_var = StringVar()

    # Create widgets
    Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=10)
    Combobox(root, textvariable=genre_var, values=unique_genres).grid(row=1, column=1, padx=10, pady=10)

    Label(root, text="Age Rating:").grid(row=2, column=0, padx=10, pady=10)
    Combobox(root, textvariable=age_rating_var, values=unique_age_ratings).grid(row=2, column=1, padx=10, pady=10)

    Label(root, text="Duration (min):").grid(row=3, column=0, padx=10, pady=10)
    Entry(root, textvariable=duration_var).grid(row=3, column=1, padx=10, pady=10)

    predict_button = Button(root, text="Predict Platform", command=lambda: validate_and_predict(root, genre_var.get(), duration_var.get(), age_rating_var.get()))
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

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

        # After prediction, close the current window and open the result window
        previous_root.destroy()
        result_window(predicted_platform, clf_regressor, genre_encoded, duration, age_rating_encoded)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Please ensure all inputs are correctly formatted.\n{e}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction.\n{e}")

# Function to create the Result window based on the platform
def result_window(platform, clf_regressor, genre_encoded, duration, age_rating_encoded):
    root = Tk()
    root.title(f"{platform} Prediction")

    platform_theme = {
        'Netflix': {
            'color': '#E50914',
            'logo_path': '/Users/samer/Documents/github_repos/Cinemalytics/images/netflix_logo_4.png'
        },
        'Prime Video': {
            'color': '#00A8E1',
            'logo_path': '/Users/samer/Documents/github_repos/Cinemalytics/images/prime_video_logo1.webp'
        },
        'Disney+': {
            'color': '#000c7c',
            'logo_path': '/Users/samer/Documents/github_repos/Cinemalytics/images/disney_plus_logo1.png'
        }
    }

    theme = platform_theme.get(platform, {'color': 'white', 'logo_path': ''})
    root.configure(bg=theme['color'])

    logo_image = load_image(theme['logo_path'])
    logo_label = Label(root, image=logo_image, bg=theme['color'])
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    message_label = Label(root, text=f"Based on your criteria, the best platform for your movie is {platform}", bg=theme['color'], fg='white')
    message_label.pack(pady=10)

    revenue_button = Button(root, text="Predict Revenue", command=lambda: revenue_window(root, clf_regressor, genre_encoded, duration, age_rating_encoded))
    revenue_button.pack(pady=10)

    root.mainloop()

# Function to create the Revenue window
def revenue_window(previous_root, clf_regressor, genre_encoded, duration, age_rating_encoded):
    previous_root.destroy()
    root = Tk()
    root.title("Revenue Prediction")

    # Prepare features for the revenue prediction
    input_features = pd.DataFrame({
        'genre_encoded': [genre_encoded],
        'duration_min': [duration],
        'age_rating_encoded': [age_rating_encoded]
    })

    # Predict the revenue
    predicted_revenue = clf_regressor.predict(input_features)[0]
    lower_estimate = predicted_revenue * 0.9
    upper_estimate = predicted_revenue * 1.1

    revenue_label = Label(root, text=f"Based on your criteria, the predicted revenue is between ${lower_estimate:,.2f} and ${upper_estimate:,.2f}")
    revenue_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_window()
