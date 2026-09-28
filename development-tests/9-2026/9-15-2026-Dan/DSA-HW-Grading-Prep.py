from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

# Change these for each assignment: HW1, HW2, HW3, etc.
ASSIGNMENT_NAME = "HW2"
SUBMISSIONS_FOLDER = "HW2submissions"

# Student -> section mapping.
# Keep this for future homeworks unless enrollment changes.
STUDENT_SECTIONS = {
    # section01
    "blackwellchris": "Section01",
    "clarkehannah": "Section01",
    "demerssthilairejerye": "Section01",
    "haughwoutavery": "Section01",
    "koullapimilena": "Section01",
    "santanaadrian": "Section01",
    "venechjoseph": "Section01",

    # section02
    "condematthew": "Section02",
    "hilatristan": "Section02",
    "youngjack": "Section02",

    # section03
    "blueparker": "Section03",
    "caballeroespinosaisrael": "Section03",
    "cicerrellaolivia": "Section03",
    "claimongk": "Section03",
    "delatesean": "Section03",
    "masuecosmavis": "Section03",
    "mcnallyjack": "Section03",
    "petraccoeddie": "Section03",
    "robertzane": "Section03",
    "rochacarson": "Section03",
    "sharberairen": "Section03",
    "singhsureena": "Section03",
    "treigerkappshochstatterclara": "Section03",
    "williamsalexander": "Section03",
    "youngjackson": "Section03",

    # section04
    "campagnacalli": "Section04",
    "curtisciara": "Section04",
    "davisjoshua": "Section04",
    "gomezmatthew": "Section04",
    "gumienygavin": "Section04",
    "kowalskirichie": "Section04",
    "milanesdayton": "Section04",
}

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / SUBMISSIONS_FOLDER
OUTPUT_DIR = BASE_DIR / ASSIGNMENT_NAME


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not SOURCE_DIR.exists():
    print("Error: submissions folder does not exist:")
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

processed_students = set()
processed_files = 0

for java_file in java_files:
    parts = java_file.stem.split("_")
    student_name = parts[0]

    # Ignore files whose student is not in the section mapping.
    section = STUDENT_SECTIONS.get(student_name)
    if section is None:
        continue

    # Restore the student's original Java filename.
    # Example:
    # santanaadrian_2061820_54292363_HW2.java -> HW2.java
    actual_filename = parts[-1] + java_file.suffix

    # These directories are only created when this student
    # actually has a submission.
    section_dir = OUTPUT_DIR / section
    student_dir = section_dir / student_name

    section_dir.mkdir(exist_ok=True)
    student_dir.mkdir(exist_ok=True)

    destination = student_dir / actual_filename

    if destination.exists():
        print(
            f"WARNING: {destination} already exists. "
            f"Overwriting it."
        )

    shutil.copy2(java_file, destination)

    processed_files += 1
    processed_students.add((section, student_name))

    print(
        f"{java_file.name}\n"
        f"  -> {ASSIGNMENT_NAME}/{section}/{student_name}/{actual_filename}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\nFinished.")
print(f"Processed {processed_files} Java file(s).")
print(f"Organized {len(processed_students)} student folder(s).")
print(f"Output directory: {OUTPUT_DIR}")
