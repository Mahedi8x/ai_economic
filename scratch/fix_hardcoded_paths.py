"""
Fix hardcoded C:\ai_economic paths in all Python files.
Adds BASE_DIR = Path(__file__).resolve().parent (or .parent.parent for scripts/) at top
and replaces all hardcoded path references.
"""
import os
import re
import glob

BASE = r"C:\ai_economic"

def fix_file(filepath):
    rel = filepath.replace(BASE + os.sep, "")
    
    # Determine depth for BASE_DIR calculation
    depth = rel.count(os.sep)
    if depth == 0:
        base_dir_expr = "Path(__file__).resolve().parent"
    elif depth == 1:
        base_dir_expr = "Path(__file__).resolve().parent.parent"
    else:
        base_dir_expr = "Path(__file__).resolve()" + ".parent" * (depth + 1)
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    
    original = txt
    
    # Skip if no hardcoded path
    if r"C:\ai_economic" not in txt and "C:/ai_economic" not in txt:
        return False
    
    # Check if pathlib already imported
    has_pathlib = "from pathlib import Path" in txt or "import pathlib" in txt
    has_base_dir = "BASE_DIR" in txt
    
    # Replace all variations of the hardcoded path
    # 1. r"C:\ai_economic" -> BASE_DIR  (as string in os.path.join etc.)
    txt = re.sub(r'r"C:\\ai_economic"', 'BASE_DIR', txt)
    txt = re.sub(r"r'C:\\\\ai_economic'", 'BASE_DIR', txt)
    txt = re.sub(r'"C:/ai_economic"', 'str(BASE_DIR)', txt)
    txt = re.sub(r"'C:/ai_economic'", 'str(BASE_DIR)', txt)
    txt = re.sub(r'"C:\\\\ai_economic"', 'str(BASE_DIR)', txt)
    txt = re.sub(r"'C:\\\\ai_economic'", 'str(BASE_DIR)', txt)
    
    # 2. os.path.join(BASE_DIR, ...) already handled, but fix str() wrapping
    # 3. f-strings with {BASE_DIR} - leave as is
    
    # Add BASE_DIR and pathlib import if needed
    if "BASE_DIR" in txt and not has_base_dir:
        base_dir_line = f"\nBASE_DIR = {base_dir_expr}\n"
        pathlib_import = "from pathlib import Path\n"
        
        # Insert after existing imports block
        # Find last import line
        lines = txt.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import_idx = i
        
        if not has_pathlib:
            lines.insert(last_import_idx + 1, pathlib_import.strip())
            last_import_idx += 1
        
        lines.insert(last_import_idx + 1, base_dir_line.strip())
        txt = "\n".join(lines)
    
    if txt == original:
        return False
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(txt)
    
    return True


fixed_files = []
skipped = []

for f in glob.glob(BASE + r"\**\*.py", recursive=True):
    if "archive_synthetic_demo" in f or "scratch" in f:
        continue
    try:
        if fix_file(f):
            fixed_files.append(f.replace(BASE + os.sep, ""))
    except Exception as e:
        skipped.append(f"{f.replace(BASE + os.sep, '')}: {e}")

print(f"\nFixed {len(fixed_files)} files:")
for ff in fixed_files:
    print(f"  ✓ {ff}")

if skipped:
    print(f"\nSkipped/errors ({len(skipped)}):")
    for s in skipped:
        print(f"  ✗ {s}")

# Final check
remaining = []
for f in glob.glob(BASE + r"\**\*.py", recursive=True):
    if "archive_synthetic_demo" in f or "scratch" in f:
        continue
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
        if r"C:\ai_economic" in txt or "C:/ai_economic" in txt:
            remaining.append(f.replace(BASE + os.sep, ""))
    except:
        pass

print(f"\nRemaining hardcoded paths: {len(remaining)}")
for r in remaining:
    print(f"  ⚠ {r}")
