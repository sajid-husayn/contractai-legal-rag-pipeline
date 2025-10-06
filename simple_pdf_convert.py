#!/usr/bin/env python3
"""
Simple text to PDF converter using fpdf2
"""
from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Legal Document', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def convert_text_to_pdf(text_file_path, pdf_file_path):
    """Convert text file to PDF"""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 10)
        
        # Split content into lines and add to PDF
        lines = content.split('\n')
        for line in lines:
            # Handle long lines by wrapping
            if len(line) > 80:
                words = line.split(' ')
                current_line = ''
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + ' '
                    else:
                        if current_line:
                            pdf.cell(0, 5, current_line.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
                        current_line = word + ' '
                if current_line:
                    pdf.cell(0, 5, current_line.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
            else:
                pdf.cell(0, 5, line.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
        
        pdf.output(pdf_file_path)
        print(f"Created PDF: {pdf_file_path}")
        return True
    except Exception as e:
        print(f"Error converting {text_file_path}: {e}")
        return False

def main():
    """Convert all text files to PDFs"""
    docs_dir = "documents"
    
    text_files = [
        "employment_agreement.txt",
        "non_disclosure_agreement.txt", 
        "service_contract.txt",
        "consulting_agreement.txt"
    ]
    
    for text_file in text_files:
        text_path = os.path.join(docs_dir, text_file)
        pdf_path = os.path.join(docs_dir, text_file.replace('.txt', '.pdf'))
        
        if os.path.exists(text_path):
            convert_text_to_pdf(text_path, pdf_path)

if __name__ == "__main__":
    main()