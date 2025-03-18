# Facialytics (Advanced face recognition system with real Time detection )
import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
import pickle
import cv2
from PIL import Image, ImageTk
import numpy as np

def fetch_faces(db_path="face_recognition.db"):
    """Retrieve all stored faces from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM faces")
    data = cursor.fetchall()
    conn.close()
    return data

def insert_face(name, embedding, db_path="face_recognition.db"):
    """Insert a new face entry into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, pickle.dumps(embedding)))
    conn.commit()
    conn.close()
    messagebox.showinfo("Success", f"Face data for {name} inserted successfully.")

def delete_face(face_id, db_path="face_recognition.db"):
    """Delete a face entry from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    messagebox.showinfo("Deleted", f"Deleted face ID {face_id} from the database.")

def browse_image():
    """Open file dialog to select an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((160, 160))
        photo = ImageTk.PhotoImage(image)
        img_label.config(image=photo)
        img_label.image = photo
        process_image(file_path)

def process_image(image_path):
    """Process the image and extract embeddings."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (160, 160))
    face_embedding = np.random.rand(128)  # Simulating embedding extraction
    name = name_entry.get()
    if name:
        insert_face(name, face_embedding)
    else:
        messagebox.showerror("Error", "Enter a name before uploading.")

def refresh_list():
    """Refresh the face list from the database."""
    face_list.delete(0, tk.END)
    for face in fetch_faces():
        face_list.insert(tk.END, f"ID: {face[0]}, Name: {face[1]}")

# GUI Setup
root = tk.Tk()
root.title("Face Database Manager")
root.geometry("500x500")

name_label = tk.Label(root, text="Enter Name:")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack()

upload_btn = tk.Button(root, text="Upload Image", command=browse_image)
upload_btn.pack()

img_label = tk.Label(root)
img_label.pack()

face_list = tk.Listbox(root, width=50)
face_list.pack()
refresh_list()

delete_btn = tk.Button(root, text="Delete Selected", command=lambda: delete_face(face_list.get(tk.ACTIVE).split(':')[1].strip()))
delete_btn.pack()

refresh_btn = tk.Button(root, text="Refresh List", command=refresh_list)
refresh_btn.pack()

root.mainloop()
