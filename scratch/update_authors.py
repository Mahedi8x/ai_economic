import os
import subprocess
import shutil
from pathlib import Path

# Update scratch/build_ieee_extended_abstract.py
p_docx_script = r"C:\ai_economic\scratch\build_ieee_extended_abstract.py"
with open(p_docx_script, "r", encoding="utf-8") as f:
    txt_docx = f.read()

old_auth_docx = 'run_auth = p_author.add_run("Md. Mehedi Hasan\\nDepartment of Computer Science & Engineering\\nOpen-Source Codebase: https://github.com/Mahedi8x/ai_economic")'
new_auth_docx = 'run_auth = p_author.add_run("Md Mohi Uddin Mahedi, Mst. Mimuna Parvin Mim\\nDepartment of Computer Science & Engineering\\nOpen-Source Codebase: https://github.com/Mahedi8x/ai_economic")'

txt_docx = txt_docx.replace(old_auth_docx, new_auth_docx)
with open(p_docx_script, "w", encoding="utf-8") as f:
    f.write(txt_docx)

# Update scratch/build_ieee_extended_abstract_pdf.py
p_pdf_script = r"C:\ai_economic\scratch\build_ieee_extended_abstract_pdf.py"
with open(p_pdf_script, "r", encoding="utf-8") as f:
    txt_pdf = f.read()

old_auth_pdf = """        <div class="author-block">
            <strong>Md. Mehedi Hasan</strong><br>
            <span style="font-size: 8.5pt; color: #475569;">
                Department of Computer Science and Engineering<br>
                Open-Source Research Repository: <u>https://github.com/Mahedi8x/ai_economic</u>
            </span>
        </div>"""

new_auth_pdf = """        <div class="author-block">
            <strong>Md Mohi Uddin Mahedi<sup>1</sup>, Mst. Mimuna Parvin Mim<sup>2</sup></strong><br>
            <span style="font-size: 8.5pt; color: #475569;">
                Department of Computer Science and Engineering<br>
                Open-Source Research Repository: <u>https://github.com/Mahedi8x/ai_economic</u>
            </span>
        </div>"""

txt_pdf = txt_pdf.replace(old_auth_pdf, new_auth_pdf)
with open(p_pdf_script, "w", encoding="utf-8") as f:
    f.write(txt_pdf)

print("Updated generator scripts with the two author names!")
