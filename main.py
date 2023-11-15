import subprocess

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import sqlite3


def apply_sharpening_filter(image):
    sharpening_filter = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]], dtype=np.float32)
    sharpened_image = cv2.filter2D(image, -1, sharpening_filter)
    return sharpened_image


def calculate_segmented_volume(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    distance_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]
    segmented_volume = np.sum(np.all(image == [0, 255, 0], axis=-1))
    return segmented_volume


def classify_image(image_path, class_average_volumes):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Calculer le volume de la partie segmentée de l'image
    segmented_volume = calculate_segmented_volume(image)

    if segmented_volume is None:
        print(f"Error: Unable to calculate segmented volume for image at {image_path}")
        return None

    # Trouver la classe la plus proche en fonction du volume
    closest_class = min(class_average_volumes, key=lambda x: abs(class_average_volumes[x] - segmented_volume))

    return closest_class


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alzheimer Classifier")

        # SQLite Database
        self.conn = sqlite3.connect('users.db')
        self.create_users_table()

        # Initial Screen
        self.initial_frame = ttk.Frame(root, padding="10")
        self.initial_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.login_button = ttk.Button(self.initial_frame, text="Login", command=self.show_login_screen)
        self.login_button.grid(row=0, column=0, pady=10)
        self.register_button = ttk.Button(self.initial_frame, text="Register", command=self.show_register_screen)
        self.register_button.grid(row=0, column=1, pady=10)

        # Registration Screen
        self.registration_frame = ttk.Frame(root, padding="10")
        self.new_username_label = ttk.Label(self.registration_frame, text="New Username:")
        self.new_username_label.grid(row=1, column=0, pady=5)
        self.new_username_entry = ttk.Entry(self.registration_frame)
        self.new_username_entry.grid(row=1, column=1, pady=5)
        self.new_password_label = ttk.Label(self.registration_frame, text="New Password:")
        self.new_password_label.grid(row=2, column=0, pady=5)
        self.new_password_entry = ttk.Entry(self.registration_frame, show="*")
        self.new_password_entry.grid(row=2, column=1, pady=5)
        self.register_button = ttk.Button(self.registration_frame, text="Register", command=self.register)
        self.register_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Login Screen
        self.login_frame = ttk.Frame(root, padding="10")
        self.username_label = ttk.Label(self.login_frame, text="Username:")
        self.username_label.grid(row=0, column=0, pady=5)
        self.username_entry = ttk.Entry(self.login_frame)
        self.username_entry.grid(row=0, column=1, pady=5)
        self.password_label = ttk.Label(self.login_frame, text="Password:")
        self.password_label.grid(row=1, column=0, pady=5)
        self.password_entry = ttk.Entry(self.login_frame, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)
        self.login_button = ttk.Button(self.login_frame, text="Login", command=self.login)
        self.login_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Image Upload and Classification Screen
        self.image_frame = ttk.Frame(root, padding="10")
        self.image_label = ttk.Label(self.image_frame, text="Please upload image (IRM):", font=("Helvetica", 14))
        self.image_label.pack()
        self.upload_button = ttk.Button(self.image_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        self.classify_button = ttk.Button(self.image_frame, text="Classify Image", command=self.classify_image,
                                          state=tk.DISABLED)
        self.classify_button.pack(pady=10)

        self.logged_in = False

    def create_users_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        self.conn.commit()

    def show_login_screen(self):
        self.clear_initial_frame()
        self.clear_registration_frame()
        self.clear_image_frame()
        self.clear_login_frame()
        self.login_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def show_register_screen(self):
        self.clear_initial_frame()
        self.clear_login_frame()
        self.clear_image_frame()
        self.registration_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def register(self):
        new_username = self.new_username_entry.get()
        new_password = self.new_password_entry.get()

        if new_username and new_password:
            cursor = self.conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (new_username, new_password))
            self.conn.commit()
            messagebox.showinfo("Registration", "Registration successful. You can now log in.")
            self.clear_registration_frame()
            self.show_login_screen()
        else:
            messagebox.showwarning("Registration Error", "Please enter a username and password.")

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username and password:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
            user = cursor.fetchone()

            if user:
                messagebox.showinfo("Login", f"Welcome, {username}!")
                self.logged_in = True
                self.show_image_frame()
                subprocess.run(["python", "doctor_app.py"])

            else:
                messagebox.showwarning("Login Failed", "Invalid username or password.")
                self.clear_login_frame()
        else:
            messagebox.showwarning("Login Error", "Please enter a username and password.")
            self.clear_login_frame()

    def upload_image(self):
        if self.logged_in:
            file_path = filedialog.askopenfilename(title="Select an Image File",
                                                   filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

            if file_path:
                self.image = Image.open(file_path)
                self.image.thumbnail((300, 300))
                self.tk_image = ImageTk.PhotoImage(self.image)

                self.image_label.config(image=self.tk_image)
                self.image_label.image = self.tk_image

                self.image_path = file_path
                self.classify_button.config(state=tk.NORMAL)
        else:
            print("Please log in first.")

    def classify_image(self):
        if hasattr(self, 'image_path') and self.logged_in:
            print(f"Classifying image: {self.image_path}")

            # Load the volumes from the CSV
            volumes_df = pd.read_csv('./volumes_moyens.csv')
            class_average_volumes = {row['Classe']: row['Volume moyen'] for index, row in volumes_df.iterrows()}

            # Classify the image
            predicted_class = classify_image(self.image_path, class_average_volumes)

            # Show the classification result
            if predicted_class is not None:
                if predicted_class in ["ModerateDemented", "VeryMildDemented"]:
                    result = 'Demented'
                else:
                    result = 'Non Demented'
                messagebox.showinfo("Classification Result", f"The image is classified as: {result}")
            else:
                messagebox.showwarning("Classification Error", "Failed to classify the image.")
        else:
            print("No image selected or not logged in.")

    def show_image_frame(self):
        self.clear_login_frame()
        self.clear_registration_frame()
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def clear_initial_frame(self):
        self.initial_frame.grid_forget()

    def clear_login_frame(self):
        self.username_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.login_frame.grid_forget()

    def clear_registration_frame(self):
        self.new_username_entry.delete(0, tk.END)
        self.new_password_entry.delete(0, tk.END)
        self.registration_frame.grid_forget()

    def clear_image_frame(self):
        self.image_label.config(image="")
        self.image_label.image = None
        self.image_path = None
        self.classify_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()