#!/usr/bin/env python3
"""
Convert text files to PDF format for RAG pipeline testing
"""
from reportlab.lib.pagesizes import letter
from reportlab.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def convert_text_to_pdf(text_file_path, pdf_file_path):
    """Convert a text file to PDF"""
    # Read the text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_file_path, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=0,
        spaceAfter=12,
    )
    
    # Build story
    story = []
    
    # Split content into lines and process
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 12))
            continue
            
        # Title detection (all caps lines)
        if line.isupper() and len(line) > 5:
            story.append(Paragraph(line, title_style))
        else:
            # Regular paragraph
            story.append(Paragraph(line, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"Created PDF: {pdf_file_path}")

def main():
    """Convert all text files in documents directory to PDFs"""
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
        else:
            print(f"File not found: {text_path}")

if __name__ == "__main__":
    main()