import pandas as pd
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, font, Frame, Canvas
from PIL import Image, ImageTk
from joblib import load
from tkinter.ttk import Combobox, Style, Button as TtkButton
import time

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

def fade_out(root):
    alpha = root.attributes("-alpha")
    while alpha > 0:
        alpha -= 0.05  # Decrease the opacity by 5%
        root.attributes("-alpha", alpha)
        root.update()
        time.sleep(0.05)  # Wait for 25ms

    root.destroy()

def fade_in(root):
    alpha = 0
    root.attributes("-alpha", alpha)
    while alpha < 1:
        alpha += 0.05  # Increase the opacity by 5%
        new_alpha = min(alpha, 0.9)  # Ensure that the max opacity is 90%
        root.attributes("-alpha", new_alpha)
        root.update()
        time.sleep(0.05)  # Wait for 25ms

# Function to center the window
def center_window(root, width=900, height=500):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - width/2)
    center_y = int(screen_height/2 - height/2)
    root.geometry(f'{width}x{height}+{center_x}+{center_y}')

def start_window():
    root = Tk()
    # root.attributes("-alpha", 0.9)
    center_window(root, width=900, height=500)
    root.title("Cinemalytics")

    canvas = Canvas(root, width=900, height=500)
    canvas.pack(fill="both", expand=True)

    # Load and place the logo image on the Canvas
    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(200, 200))
    # Slightly lower the logo's position on the canvas
    canvas.create_image(450, 150, image=logo_image, anchor='center')  # Adjusted position

    # Define a larger font for the project title
    project_title_font = font.Font(size=24, family='Helvetica', weight='bold')

    # Slightly lower the project title's position below the logo on the canvas
    project_title_label = Label(root, text="Cinemalytics: Empowering Your Vision with Insights", font=project_title_font)
    # Adjusted position
    project_title_window = canvas.create_window(450, 350, window=project_title_label, anchor='center')

    # Style for the Start button
    style = Style()
    style.configure("TButton", borderwidth=1, bordercolor="black", background="#d9d9d9", font=('Helvetica', 14), padding=6)

    # Place the Start button at the bottom
    start_button = TtkButton(root, text="Start", command=lambda: platform_window(root), style="TButton")
    start_button_window = canvas.create_window(450, 450, window=start_button, anchor='center')  # Position is good

    # Keep references to the images and widgets to prevent garbage collection
    canvas.logo_image = logo_image
    canvas.project_title_label = project_title_label

    fade_in(root)
    root.mainloop()


def platform_window(previous_root):
    previous_root.destroy()
    root = Tk()
    root.attributes("-alpha", 0.9)
    center_window(root)
    root.title("Platform Prediction")

    # Configure the style for the TButton to be grey
    style = Style()
    style.configure("TButton", borderwidth=1, bordercolor="black", background="#d9d9d9", font=('Helvetica', 14), padding=6)

    logo_image = load_image('/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png', size=(200, 200))
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
    predict_button = TtkButton(root, text="Predict Platform", command=lambda: validate_and_predict(root, genre_var.get(), duration_var.get(), age_rating_var.get()), style="TButton")
    predict_button.grid(row=4, column=0, columnspan=2, pady=10)

    for i in range(5):
        root.grid_rowconfigure(i, weight=1)
    for j in range(2):
        root.grid_columnconfigure(j, weight=1)

    fade_in(root)
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
    center_window(root, width=900, height=500)
    root.title(f"{platform} Prediction")

    # Define font styles
    large_font = font.Font(size=16, family='Helvetica')  # Normal font
    bold_font = font.Font(size=16, family='Helvetica', weight='bold')  # Bold font for labels

    # Configure the grid
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0, minsize=200)  # This column will contain the logo
    root.grid_columnconfigure(2, weight=3)  # This column will contain the text message

    app_logo_path = '/Users/samer/Documents/github_repos/Cinemalytics/images/cinemalytics_nobackground.png'
    app_logo_image = load_image(app_logo_path, size=(150, 150))
    app_logo_label = Label(root, image=app_logo_image)
    app_logo_label.image = app_logo_image
    app_logo_label.grid(row=0, column=0, columnspan=3, pady=20)

    platform_logos = {
        'Netflix': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_netflix_shadow.png',
        'Prime Video': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_prime_video_shadow.png',
        'Disney+': '/Users/samer/Documents/github_repos/Cinemalytics/images/thumbnail_disney_plus_shadow.png',
    }

    # Add an invisible frame to push the logo towards the center
    invisible_frame = Frame(root)
    invisible_frame.grid(row=1, column=0, sticky='nsew')

    logo_path = platform_logos.get(platform)
    if logo_path:
        logo_image = load_image(logo_path, size=(200, 200))
        logo_label = Label(root, image=logo_image)
        logo_label.image = logo_image
        logo_label.grid(row=1, column=1, padx=20, sticky='nsew')  # Adjust the padx value as needed

    message_text = (
        f"Genre: {genre}\n\nAge Rating: {age_rating}\n\nDuration: {duration} minutes\n\n"
        f"Recommendation: {platform}\n\nPredicted Revenue: ${predicted_revenue:,.2f}"
    )
    message_label = Label(root, text=message_text, font=large_font, justify='left', anchor='n')
    message_label.grid(row=1, column=2, padx=(20, 20), pady=(20, 20), sticky='nsew')  # Adjust the padx and pady values as needed

    # Apply the styled 'TButton' to the back button
    back_button = TtkButton(root, text="Back to Start", command=lambda: restart_app(root), style="TButton")
    back_button.grid(row=2, column=0, columnspan=3, pady=(20, 40))  # Increase pady as needed for spacing

    fade_in(root)
    root.mainloop()


def restart_app(current_root):
    fade_out(current_root)
    # current_root.destroy()
    start_window()


if __name__ == "__main__":
    start_window()
