import pandas as pd
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Canvas, font, LEFT
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

def load_image(path, size=None):
    image = Image.open(path)
    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


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
def center_window(root, width=900, height=650):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - width/2)
    center_y = int(screen_height/2 - height/2)
    root.geometry(f'{width}x{height}+{center_x}+{center_y}')

def start_window():
    root = Tk()
    root.attributes("-alpha", 0.9)
    center_window(root, width=900, height=650)
    root.title("Cinemalytics")

    # Create a Canvas and add a background image
    canvas = Canvas(root, width=900, height=650)
    canvas.pack(fill="both", expand=True)
    background_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/window_background.png')
    # Add image to Canvas
    canvas.create_image(0, 0, image=background_image, anchor='nw')

    # Instead of using Label for the logo, use the Canvas
    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(250, 250))
    # Add logo to Canvas
    canvas.create_image(450, 175, image=logo_image, anchor='center')

    # For buttons and other widgets, use the Canvas to create window-like effects
    start_button = Button(root, text="Start", command=lambda: platform_window(root))
    start_button_window = canvas.create_window(450, 400, window=start_button, anchor='center')

    # Keep a reference to the images to prevent garbage collection
    canvas.image = background_image
    canvas.logo_image = logo_image

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

from tkinter import Tk, Label, Button, font

def result_window(platform, predicted_revenue, genre, age_rating, duration):
    root = Tk()
    root.attributes("-alpha", 0.9)
    # The center_window function should be defined elsewhere in your code
    center_window(root, width=900, height=650)
    root.title(f"{platform} Prediction")

    # Define font styles
    large_font = font.Font(size=16, family='Helvetica')  # Increase size as needed

    # Configure the grid
    root.grid_columnconfigure(0, weight=1, minsize=300)  # Adjust minsize as needed for the logo
    root.grid_columnconfigure(1, weight=3)  # This column will contain the text message

    app_logo_path = '/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png'
    app_logo_image = load_image(app_logo_path, size=(150, 150))
    app_logo_label = Label(root, image=app_logo_image)
    app_logo_label.image = app_logo_image
    app_logo_label.grid(row=0, column=0, columnspan=2, pady=20)

    platform_logos = {
        'Netflix': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_netflix_shadow.png',
        'Prime Video': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_prime_video_shadow.png',
        'Disney+': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_disney_plus_shadow.png',
    }

    logo_path = platform_logos.get(platform)
    if logo_path:
        logo_image = load_image(logo_path, size=(200, 200))
        logo_label = Label(root, image=logo_image)
        logo_label.image = logo_image
        # Adjust padx to center the logo
        logo_label.grid(row=1, column=0, padx=(20, 100), pady=20, sticky='w')  # Adjust the tuple values as needed

    # Create a text variable to adjust line spacing
    message_text = f"Genre: {genre}\n\nAge Rating: {age_rating}\n\nDuration: {duration} minutes\n\nRecommendation: {platform}\n\nPredicted Revenue: ${predicted_revenue:,.2f}"
    message_label = Label(root, text=message_text, font=large_font, justify='left', anchor='n', bg='black', fg='white')
    message_label.grid(row=1, column=1, padx=(100, 20), pady=20, sticky='nsew')  # Adjust the tuple values as needed

    # Apply the large font to the back button and increase the pady for spacing
    back_button = Button(root, text="Back to Start", command=lambda: restart_app(root), font=large_font)
    back_button.grid(row=2, column=0, columnspan=2, pady=(20, 40))  # Increase pady as needed for spacing

    root.mainloop()

def restart_app(current_root):
    current_root.destroy()
    start_window()


if __name__ == "__main__":
    start_window()
