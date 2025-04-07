# create_test_db.py

# This script creates a test SQLite database with a sample table and some data. so that i can test the sql agent on it.
# Sql agent can be used with any sql database like postges, mysql, sqlite etc.
import sqlite3

def create_and_populate_db(db_name="my_database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1) Create a sample table named "employees" with simple columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            salary INTEGER,
            department TEXT
        )
    """)

    # 2) Insert some test data
    sample_data = [
        ("Alice", 60000, "Engineering"),
        ("Bob", 55000, "Sales"),
        ("Charlie", 70000, "Engineering"),
        ("Diana", 52000, "Marketing"),
        ("Ethan", 75000, "Finance")
    ]

    cursor.executemany("""
        INSERT INTO employees (name, salary, department)
        VALUES (?, ?, ?)
    """, sample_data)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_and_populate_db()
    print("Test SQLite database created and populated successfully!")
