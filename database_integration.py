# Facialytics (Advanced face recognition system with real Time detection )
import sqlite3
import os
import numpy as np
import pickle

def initialize_database(db_path="face_recognition.db"):
    """
    Create and initialize the database with necessary tables.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def insert_face(name, embedding, db_path="face_recognition.db"):
    """
    Insert a new face entry into the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, pickle.dumps(embedding)))
    conn.commit()
    conn.close()
    print(f"Face data for {name} inserted successfully.")

def fetch_faces(db_path="face_recognition.db"):
    """
    Retrieve all face records from the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM faces")
    data = [(row[0], row[1], pickle.loads(row[2])) for row in cursor.fetchall()]
    conn.close()
    return data

def update_face_name(face_id, new_name, db_path="face_recognition.db"):
    """
    Update the name of a stored face entry.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE faces SET name = ? WHERE id = ?", (new_name, face_id))
    conn.commit()
    conn.close()
    print(f"Updated face ID {face_id} to new name: {new_name}.")

def delete_face(face_id, db_path="face_recognition.db"):
    """
    Delete a face entry from the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    print(f"Deleted face ID {face_id} from the database.")

if __name__ == "__main__":
    initialize_database()
    # Example usage
    sample_embedding = np.random.rand(128)  # Simulated embedding
    insert_face("John Doe", sample_embedding)
    print(fetch_faces())
