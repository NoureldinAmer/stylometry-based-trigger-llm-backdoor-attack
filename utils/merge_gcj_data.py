import os
import sqlite3
import glob
from pathlib import Path


def merge_sqlar_files(base_dir, output_file):
    """
    Merge multiple solutions.sqlar files into a single output file
    using only name and data columns

    Args:
        base_dir: Base directory containing subdirectories with solutions.sqlar files
        output_file: Path to the output merged file
    """
    # Remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Create and initialize the output database with only name and data columns
    output_conn = sqlite3.connect(output_file)
    output_cursor = output_conn.cursor()
    output_cursor.execute(
        "CREATE TABLE IF NOT EXISTS sqlar(name TEXT PRIMARY KEY, data BLOB);")
    output_conn.commit()

    # Find all solutions.sqlar files
    sqlar_files = glob.glob(os.path.join(
        base_dir, "**", "solutions.sqlar"), recursive=True)

    print(f"Found {len(sqlar_files)} solutions.sqlar files")

    # Process each solutions.sqlar file
    for file_path in sqlar_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        print(f"Processing {file_path} (from {folder_name})...")

        try:
            # Connect to the source database
            source_conn = sqlite3.connect(file_path)
            source_cursor = source_conn.cursor()

            # Get only name and data from the source database as requested
            source_cursor.execute("SELECT name, data FROM sqlar")
            entries = source_cursor.fetchall()

            # Add prefix to the name field and insert into the output database
            for entry in entries:
                name, data = entry

                output_cursor.execute(
                    "INSERT OR IGNORE INTO sqlar (name, data) VALUES (?, ?)",
                    (name, data)
                )

            # Commit changes and close the source connection
            output_conn.commit()
            source_conn.close()

        except sqlite3.Error as e:
            print(f"Error processing {file_path}: {e}")

    # Close the output connection
    output_conn.close()
    print(f"Merged file created at {output_file}")


if __name__ == "__main__":
    # Set your base directory and output file path
    base_dir = "gcj-archive-2022"
    output_file = "combined_solutions_2022.sqlar"

    merge_sqlar_files(base_dir, output_file)
