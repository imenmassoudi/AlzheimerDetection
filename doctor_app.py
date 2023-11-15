import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import pandas as pd
import sqlite3
from main import apply_sharpening_filter, calculate_segmented_volume, classify_image

class Patient:
    def __init__(self, patient_id, name, details, image_path, volume_id=None):
        self.patient_id = patient_id
        self.name = name
        self.details = details
        self.image_path = image_path
        self.volume_id = volume_id

class PatientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Patient Management System")
        self.root.geometry("800x600")  # Set the initial size
        self.center_window()  # Center the window on the screen
        self.db_connection = sqlite3.connect("patients.db")
        self.create_table()
        self.patients = []
        self.create_widgets()

    def center_window(self):
        # Center the window on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 800  # Adjust the width as needed
        window_height = 600  # Adjust the height as needed

        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    def create_table(self):
        cursor = self.db_connection.cursor()

        # Check if patients table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        patients_table_exists = cursor.fetchone()

        # Check if volumes table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='volumes'")
        volumes_table_exists = cursor.fetchone()

        if not patients_table_exists:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    details TEXT,
                    image_path TEXT,
                    volume_id INTEGER,
                    FOREIGN KEY (volume_id) REFERENCES volumes (id)
                )
            ''')

        if not volumes_table_exists:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    predicted_class TEXT,
                    volume REAL,  -- Change 'DOUBLE' to 'REAL'
                    date TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')

        self.db_connection.commit()

    def create_widgets(self):
        # Listbox
        # Title
        title_label = tk.Label(self.root, text="ALZHEIMER PATIENTS", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Listbox
        self.listbox = tk.Listbox(self.root, width=50, height=20, font=("Helvetica", 12))
        self.listbox.pack(pady=10)

        # Buttons
        self.add_button = tk.Button(self.root, text="Add Patient", command=self.add_patient, font=("Helvetica", 12))
        self.add_button.pack()
        self.delete_button = tk.Button(self.root, text="Delete Patient", command=self.delete_patient, font=("Helvetica", 12))
        self.delete_button.pack()
        self.view_button = tk.Button(self.root, text="View Details", command=self.view_details, font=("Helvetica", 12))
        self.view_button.pack()
        self.return_button = tk.Button(self.root, text="Return to List", command=self.return_to_list, font=("Helvetica", 12))
        self.return_button.pack()
        self.return_button.pack_forget()
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.show_detect_alzheimer_interface, font=("Helvetica", 12))
        self.upload_button.pack()

        # Image path and selected index
        self.selected_index = None
        self.image_path = None

        # Update the listbox with existing data
        self.update_listbox()

    def add_patient(self):
        name = simpledialog.askstring("Input", "Enter patient name:")
        if name:
            details = simpledialog.askstring("Input", f"Enter details for {name}:")
            cursor = self.db_connection.cursor()
            cursor.execute('INSERT INTO patients (name, details, image_path, volume_id) VALUES (?, ?, ?, NULL)',
                           (name, details, ''))
            self.db_connection.commit()
            self.update_listbox()

    def delete_patient(self):
        if self.selected_index is not None:
            patient_id = self.patients[self.selected_index].patient_id
            cursor = self.db_connection.cursor()
            cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
            self.db_connection.commit()
            self.selected_index = None
            self.update_listbox()

    def get_patient_volumes(self, patient_id):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT id, volume, predicted_class FROM volumes WHERE patient_id = ?', (patient_id,))
        result = cursor.fetchall()
        return result

    def print_volume_table(self):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT * FROM volumes')
        rows = cursor.fetchall()
        print("\nContents of the 'volumes' table:")
        for row in rows:
            print(row)


    def view_details(self):
        if self.selected_index is not None:
            patient = self.patients[self.selected_index]
            self.clear_widgets()

            details_frame = tk.Frame(self.root)
            details_frame.pack(expand=True, fill='both')

            details_label = tk.Label(details_frame, text=f"Details for {patient.name}:\n{patient.details}", font=("Helvetica", 14))
            details_label.pack(pady=10)

            # Display volume information
            volume_label = tk.Label(details_frame, text=f"Volumes:", font=("Helvetica", 14))
            volume_label.pack(pady=10)

            # Fetch patient_id based on the patient's name
            patient_id = self.get_patient_id(patient.name)

            if patient_id is not None:
                volumes = self.get_patient_volumes(patient_id)
                print(f"Volumes for {patient.name}: {volumes}")
                for volume in volumes:
                    volume_id, volume_data, cl = volume
                    volume_label = tk.Label(details_frame, text=f"  - Volume ID {volume_id}: {volume_data} ({cl})", font=("Helvetica", 12))
                    volume_label.pack()

            else:
                print(f"No patient found with the name: {patient.name}")

            # Display image if available
            if patient.image_path:
                image = Image.open(patient.image_path)
                image = image.resize((300, 300))
                photo = ImageTk.PhotoImage(image)

                image_label = tk.Label(details_frame, image=photo)
                image_label.image = photo
                image_label.pack(pady=10)

            return_button = tk.Button(self.root, text="Return to List", command=self.return_to_list, font=("Helvetica", 12))
            return_button.pack(pady=10)

    def get_patient_id(self, patient_name):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT id FROM patients WHERE name = ?', (patient_name,))
        result = cursor.fetchone()
        return result[0] if result else None

    def return_to_list(self):
        self.clear_widgets()
        self.listbox.pack()
        self.add_button.pack()
        self.delete_button.pack()
        self.view_button.pack()
        self.upload_button.pack()
        self.return_button.pack_forget()

        # If image-related widgets were forgotten, show the upload_button
        if not hasattr(self, 'image_path'):
            self.upload_button.pack()

    def show_detect_alzheimer_interface(self):
        self.clear_widgets()
        detect_alzheimer_label = tk.Label(self.root, text="Detect Alzheimer's", font=("Helvetica", 16, "bold"))
        detect_alzheimer_label.pack()
        upload_image_button = tk.Button(self.root, text="Upload Image",
                                        command=lambda: self.upload_image(self.selected_index), font=("Helvetica", 14))
        upload_image_button.pack()

    def upload_image(self, patient_index):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.image_path = file_path
            self.selected_index = patient_index  # Set the selected_index based on the parameter
            self.show_classify_image_interface(patient_index)

    def classify_image(self, patient_id):
        patient_id = patient_id + 1
        if hasattr(self, 'image_path'):
            print(f"Classifying image: {self.image_path}")
            print("******* ", patient_id)
            # Load the volumes from the CSV
            volumes_df = pd.read_csv('./volumes_moyens.csv')
            class_average_volumes = {row['Classe']: row['Volume moyen'] for index, row in volumes_df.iterrows()}

            # Read the image
            original_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

            # Apply sharpening filter
            sharpened_image = apply_sharpening_filter(original_image)

            # Calculate the segmented volume on the sharpened image
            segmented_volume = calculate_segmented_volume(sharpened_image)

            # Classify the image
            predicted_class = classify_image(self.image_path, class_average_volumes)

            # Show the classification result
            if predicted_class is not None:
                if predicted_class in ["ModerateDemented", "VeryMildDemented"]:
                    result = 'Demented'
                else:
                    result = 'Non Demented'
                messagebox.showinfo("Classification Result", f"The image is classified as: {result}")
                print_button = tk.Button(self.root, text="Print Volumes Table", command=self.print_volume_table, font=("Helvetica", 12))
                print_button.pack()
                cursor = self.db_connection.cursor()
                print("offffff", patient_id, predicted_class, segmented_volume, self.get_current_date())
                cursor.execute(
                    'INSERT INTO volumes (patient_id,predicted_class, volume, date) VALUES (?, ?, ?, ?)',
                    (patient_id, predicted_class, float(segmented_volume), self.get_current_date()))

                # Update the 'volume_id' in the 'patients' table
                volume_id = cursor.lastrowid
                cursor.execute('UPDATE patients SET volume_id = ? WHERE id = ?', (volume_id, patient_id))

                self.db_connection.commit()
            else:
                messagebox.showwarning("Classification Error", "Failed to classify the image.")
        else:
            print("No image selected.")

    def show_classify_image_interface(self, patient_index):
        self.clear_widgets()

        image = Image.open(self.image_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)

        image_label = tk.Label(self.root, image=photo)
        image_label.image = photo
        image_label.pack()

        classify_button = tk.Button(self.root, text="Classify Image", command=lambda: self.classify_image(patient_index), font=("Helvetica", 14))
        classify_button.pack()

    def clear_widgets(self):
        # Clear all widgets from the root window
        for widget in self.root.winfo_children():
            widget.pack_forget()

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT * FROM patients')
        rows = cursor.fetchall()
        self.patients = [Patient(row[0], row[1], row[2], row[3], row[4]) for row in rows]
        if not self.patients:
            self.listbox.insert(0, "No records")
        else:
            for i, patient in enumerate(self.patients, 1):
                self.listbox.insert(tk.END, f"{i}. {patient.name}")

    def on_select(self, event):
        selected_indices = self.listbox.curselection()
        if selected_indices:
            self.selected_index = selected_indices[0]
        else:
            self.selected_index = None

    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    root = tk.Tk()
    app = PatientApp(root)
    app.listbox.bind('<<ListboxSelect>>', app.on_select)
    root.mainloop()
