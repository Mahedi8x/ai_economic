"""
Render full academic paper manuscript to publication-grade PDF using headless Microsoft Edge.
Produces an IEEE/Elsevier-standard academic layout.
"""
import os
import subprocess
import markdown

BASE_DIR = r"C:\ai_economic"
MD_PATH = os.path.join(BASE_DIR, "PAPER_MANUSCRIPT.md")
HTML_PATH = os.path.join(BASE_DIR, "Bangladesh_Economic_AI_Research_Paper.html")
PDF_PATH = os.path.join(BASE_DIR, "Bangladesh_Economic_AI_Research_Paper.pdf")
DESKTOP_PDF = os.path.join(r"C:\Users\Mehedi\Desktop", "Bangladesh_Economic_AI_Research_Paper.pdf")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

JOURNAL_CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #718096;
    }
}

body {
    font-family: 'Times New Roman', Times, 'Cambria', Georgia, serif;
    color: #1a202c;
    line-height: 1.6;
    font-size: 10pt;
    margin: 0;
    padding: 0;
    background: #ffffff;
}

h1 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #0f2d59;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 12px;
    line-height: 1.3;
}

blockquote {
    border-left: 3px solid #0f2d59;
    background: #f8fafc;
    padding: 8px 14px;
    margin: 10px 0 16px 0;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 9pt;
    color: #4a5568;
    line-height: 1.5;
}

h2 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 12pt;
    font-weight: 700;
    color: #0f2d59;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

h3 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: 600;
    color: #1a4278;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 9px;
    text-align: justify;
    text-justify: inter-word;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    text-align: left;
}

th {
    background-color: #0f2d59;
    color: #ffffff;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}

code {
    font-family: Consolas, 'Courier New', monospace;
    font-size: 8.5pt;
    background-color: #edf2f7;
    padding: 1px 4px;
    border-radius: 3px;
}

pre {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 10px;
    border-radius: 4px;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
}

li {
    margin-bottom: 3px;
    text-align: justify;
}

@media print {
    h1, h2, h3 {
        page-break-after: avoid;
    }
    table, pre, blockquote {
        page-break-inside: avoid;
    }
}
"""

def generate_paper_pdf():
    print("=" * 70)
    print(" [*] GENERATING FULL ACADEMIC RESEARCH PAPER PDF")
    print("=" * 70)
    
    with open(MD_PATH, "r", encoding="utf-8", errors="ignore") as f:
        md_text = f.read()
        
    html_content = markdown.markdown(md_text, extensions=['extra', 'tables', 'toc', 'nl2br'])
    
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Academic Research Paper - Mahedi Hasan</title>
    <style>
        {JOURNAL_CSS}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[1/2] Generated HTML: {HTML_PATH}")
    
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(PDF_PATH):
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"[2/2] Generated PDF: {PDF_PATH} ({size_kb:.1f} KB)")
        
        # Copy to Desktop
        import shutil
        shutil.copy2(PDF_PATH, DESKTOP_PDF)
        print(f" [+] Copied to Desktop: {DESKTOP_PDF}")
    else:
        print(f"Failed to generate PDF: {res.stderr}")

if __name__ == "__main__":
    generate_paper_pdf()
