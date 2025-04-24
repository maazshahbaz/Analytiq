# create_test_db.py

import sqlite3

def create_and_populate_db(db_name="my_database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # ------------------------------------
    # 1. Create the employees table (administrative staff)
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            salary INTEGER,
            department TEXT
        )
    """)

    employees_data = [
        ("Alice", 60000, "Administration"),
        ("Bob", 55000, "Administration"),
        ("Charlie", 70000, "Maintenance"),
        ("Diana", 52000, "Marketing"),
        ("Ethan", 75000, "Finance"),
        ("Fiona", 68000, "Human Resources"),
        ("George", 62000, "Engineering"),
        ("Hannah", 58000, "Sales"),
        ("Ian", 71000, "Engineering"),
        ("Julia", 64000, "Marketing")
    ]
    cursor.executemany("""
        INSERT INTO employees (name, salary, department)
        VALUES (?, ?, ?)
    """, employees_data)

    # ------------------------------------
    # 2. Create the departments table
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            building TEXT,
            phone TEXT
        )
    """)

    departments_data = [
        ("Engineering", "Tech Building", "555-1001"),
        ("Business", "Commerce Hall", "555-1002"),
        ("Humanities", "Arts Center", "555-1003"),
        ("Sciences", "Science Complex", "555-1004"),
        ("Social Sciences", "Social Hall", "555-1005"),
        ("Administration", "Main Office", "555-1000")
    ]
    cursor.executemany("""
        INSERT INTO departments (name, building, phone)
        VALUES (?, ?, ?)
    """, departments_data)

    # ------------------------------------
    # 3. Create the faculty table
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department_id INTEGER,
            title TEXT,
            salary INTEGER,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    faculty_data = [
        ("Dr. Smith", 1, "Professor", 95000),
        ("Dr. Johnson", 1, "Associate Professor", 85000),
        ("Dr. Williams", 2, "Professor", 98000),
        ("Dr. Brown", 3, "Assistant Professor", 75000),
        ("Dr. Jones", 4, "Professor", 105000),
        ("Dr. Miller", 5, "Associate Professor", 88000)
    ]
    cursor.executemany("""
        INSERT INTO faculty (name, department_id, title, salary)
        VALUES (?, ?, ?, ?)
    """, faculty_data)

    # ------------------------------------
    # 4. Create the students table
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            department_id INTEGER,
            enrollment_year INTEGER,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    students_data = [
        ("Alex", 19, 1, 2021),
        ("Bella", 20, 2, 2020),
        ("Chris", 21, 3, 2019),
        ("Diana", 22, 4, 2021),
        ("Evan", 18, 1, 2022),
        ("Faith", 20, 5, 2020),
        ("George", 23, 2, 2018),
        ("Holly", 19, 1, 2021),
        ("Ian", 21, 4, 2019),
        ("Jenna", 20, 5, 2020)
    ]
    cursor.executemany("""
        INSERT INTO students (name, age, department_id, enrollment_year)
        VALUES (?, ?, ?, ?)
    """, students_data)

    # ------------------------------------
    # 5. Create the courses table
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            name TEXT,
            department_id INTEGER,
            credits INTEGER,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    courses_data = [
        ("ENG101", "Introduction to Engineering", 1, 3),
        ("BUS201", "Principles of Marketing", 2, 4),
        ("HUM301", "Modern Literature", 3, 3),
        ("SCI401", "Physics II", 4, 4),
        ("SOC101", "Introduction to Sociology", 5, 3),
        ("ENG202", "Thermodynamics", 1, 4)
    ]
    cursor.executemany("""
        INSERT INTO courses (code, name, department_id, credits)
        VALUES (?, ?, ?, ?)
    """, courses_data)

    # ------------------------------------
    # 6. Create the enrollments table
    # ------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            course_id INTEGER,
            grade TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(course_id) REFERENCES courses(id)
        )
    """)

    enrollments_data = [
        (1, 1, "A"),
        (2, 2, "B+"),
        (3, 3, "A-"),
        (4, 4, "B"),
        (5, 1, "B+"),
        (6, 5, "A"),
        (7, 6, "A-"),
        (8, 1, "B"),
        (9, 4, "B+"),
        (10, 3, "A")
    ]
    cursor.executemany("""
        INSERT INTO enrollments (student_id, course_id, grade)
        VALUES (?, ?, ?)
    """, enrollments_data)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_and_populate_db()
    print("Test SQLite database created and populated successfully!")