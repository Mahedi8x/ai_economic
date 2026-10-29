import os

base = r"C:\ai_economic"

# Fix notebook generator files: replace path in NotebookClient
nb_files = [
    "generate_commodity_notebook.py",
    "generate_food_inflation_notebook.py",
    "generate_modeling_notebook.py",
    "generate_notebook_06.py",
]

old_pat = r"r'C:\ai_economic'"
new_pat = "str(BASE_DIR)"

for fname in nb_files:
    fpath = os.path.join(base, fname)
    txt = open(fpath, encoding="utf-8", errors="ignore").read()
    new_txt = txt.replace(old_pat, new_pat)
    if new_txt != txt:
        open(fpath, "w", encoding="utf-8").write(new_txt)
        print(f"Fixed: {fname}")
    else:
        print(f"No match in {fname}")
        for i, l in enumerate(txt.split("\n")):
            if "ai_economic" in l.lower():
                print(f"  Line {i+1}: {l.strip()[:120]}")

# Fix generate_research_summary_pdf.py
fname = "generate_research_summary_pdf.py"
fpath = os.path.join(base, fname)
txt = open(fpath, encoding="utf-8", errors="ignore").read()
new_txt = txt.replace(
    r'r"C:\ai_economic\Bangladesh_Economic_AI_Research_Report.pdf"',
    'str(BASE_DIR / "Bangladesh_Economic_AI_Research_Report.pdf")'
)
if new_txt != txt:
    open(fpath, "w", encoding="utf-8").write(new_txt)
    print(f"Fixed: {fname}")

# Fix scripts/polish_paper_manuscript.py
fname = os.path.join("scripts", "polish_paper_manuscript.py")
fpath = os.path.join(base, fname)
txt = open(fpath, encoding="utf-8", errors="ignore").read()
new_txt = txt.replace(
    r'r"C:\ai_economic\PAPER_MANUSCRIPT.md"',
    'str(BASE_DIR / "PAPER_MANUSCRIPT.md")'
)
if new_txt != txt:
    open(fpath, "w", encoding="utf-8").write(new_txt)
    print(f"Fixed: {fname}")
else:
    print(f"No match in {fname}")

print("\nDone.")

# Final check
import glob
remaining = []
for f in glob.glob(base + r"\**\*.py", recursive=True):
    if "archive_synthetic_demo" in f or "scratch" in f:
        continue
    try:
        txt2 = open(f, encoding="utf-8", errors="ignore").read()
        if r"C:\ai_economic" in txt2 or "C:/ai_economic" in txt2:
            remaining.append(f.replace(base + os.sep, ""))
    except:
        pass
print(f"Remaining hardcoded paths: {len(remaining)}")
for r in remaining:
    print(f"  {r}")
