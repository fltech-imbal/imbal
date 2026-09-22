from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

# Change for each assignment: HW1, HW2, HW3, etc.
ASSIGNMENT_NAME = "HW1"

# Folder containing the raw downloaded Java submissions
SUBMISSIONS_FOLDER = "HW1submissions"

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Input/output directories
SOURCE_DIR = BASE_DIR / SUBMISSIONS_FOLDER
OUTPUT_DIR = BASE_DIR / ASSIGNMENT_NAME


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not SOURCE_DIR.exists():
    print(f"Error: submissions folder does not exist:")
    print(SOURCE_DIR)
    raise SystemExit(1)

if not SOURCE_DIR.is_dir():
    print(f"Error: {SOURCE_DIR} is not a directory.")
    raise SystemExit(1)


# ============================================================
# CREATE ASSIGNMENT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FIND JAVA FILES
# ============================================================

java_files = sorted(SOURCE_DIR.glob("*.java"))

if not java_files:
    print(f"No .java files found in: {SOURCE_DIR}")
    raise SystemExit(1)

print(f"Found {len(java_files)} Java file(s).\n")


# ============================================================
# PROCESS SUBMISSIONS
# ============================================================

for java_file in java_files:

    # --------------------------------------------------------
    # Get student name
    #
    # Example:
    # santanaadrian_2061820_54292363_HW1.java
    # ->
    # santanaadrian
    # --------------------------------------------------------

    parts = java_file.stem.split("_")

    student_name = parts[0]


    # --------------------------------------------------------
    # Get actual Java filename
    #
    # The part after the LAST underscore is treated as the
    # student's original filename.
    #
    # Examples:
    #
    # santanaadrian_2061820_54292363_HW1.java
    # -> HW1.java
    #
    # santanaadrian_2061820_54292363_SinglyLinkedList.java
    # -> SinglyLinkedList.java
    # --------------------------------------------------------

    actual_filename = parts[-1] + java_file.suffix


    # --------------------------------------------------------
    # Create student directory
    # --------------------------------------------------------

    student_dir = OUTPUT_DIR / student_name
    student_dir.mkdir(exist_ok=True)


    # --------------------------------------------------------
    # Copy file while restoring its original filename
    # --------------------------------------------------------

    destination = student_dir / actual_filename

    if destination.exists():
        print(
            f"WARNING: {destination} already exists. "
            f"Overwriting it."
        )

    shutil.copy2(java_file, destination)

    print(
        f"{java_file.name}\n"
        f"  -> {ASSIGNMENT_NAME}/{student_name}/{actual_filename}"
    )


# ============================================================
# FINISHED
# ============================================================

student_dirs = [
    directory
    for directory in OUTPUT_DIR.iterdir()
    if directory.is_dir()
]

print("\nFinished.")
print(f"Processed {len(java_files)} Java file(s).")
print(f"Created/found {len(student_dirs)} student folder(s).")
print(f"Output directory: {OUTPUT_DIR}")
