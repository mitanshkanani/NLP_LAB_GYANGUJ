# src/preprocessing/pdf_to_text_all.py

import fitz  # PyMuPDF
import os
import re
from pathlib import Path
import pytesseract
from PIL import Image
import io
import sys
import subprocess

def find_tessdata_dir():
    """
    Try to automatically find the Tesseract tessdata directory
    """
    # Common paths to check
    possible_paths = [
        # Windows
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tessdata".format(os.getenv('USERNAME')),
        
        # macOS
        "/usr/local/share/tessdata",
        
        # Linux
        "/usr/share/tesseract-ocr/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
    ]
    
    # Check if TESSDATA_PREFIX environment variable is set
    if 'TESSDATA_PREFIX' in os.environ:
        possible_paths.insert(0, os.environ['TESSDATA_PREFIX'])
    
    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it contains language files
            if any(f.endswith('.traineddata') for f in os.listdir(path)):
                return path
    
    return None

# Try to find tessdata directory
tessdata_dir = find_tessdata_dir()
if tessdata_dir:
    print(f"Found tessdata directory: {tessdata_dir}")
    os.environ['TESSDATA_PREFIX'] = tessdata_dir
else:
    print("Could not find tessdata directory automatically.")
    print("Please install Tesseract and set the TESSDATA_PREFIX environment variable.")
    sys.exit(1)

# Paths
TEXTBOOK_DIR = "data/textbooks"
PROCESSED_DIR = "data/processed"

# Create processed folder if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF, with special handling for Gujarati text
    """
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try to extract text directly first
        page_text = page.get_text("text")
        
        # If little text was extracted, try OCR for scanned documents
        if len(page_text.strip()) < 100:  # Arbitrary threshold
            print(f"   Page {page_num+1}: Using OCR for text extraction")
            try:
                # Convert PDF page to image
                mat = fitz.Matrix(2, 2)  # Higher resolution for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                # Use PIL to open image and perform OCR
                image = Image.open(io.BytesIO(img_data))
                
                # Use Tesseract with Gujarati language support
                page_text = pytesseract.image_to_string(image, lang='guj')
            except Exception as e:
                print(f"   OCR failed for page {page_num+1}: {e}")
                # Fall back to the original text extraction
                page_text = page.get_text("text")
        
        text += page_text + "\n\n"
    
    doc.close()
    return text

def normalize_gujarati_text(text):
    """
    Normalize Gujarati text by fixing common encoding issues
    and removing unnecessary whitespace.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Loop through class folders (class11, class12)
for class_folder in ["class11", "class12"]:
    class_path = os.path.join(TEXTBOOK_DIR, class_folder)
    
    if not os.path.exists(class_path):
        print(f"[WARN] Folder not found: {class_path}")
        continue
    
    for pdf_file in os.listdir(class_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(class_path, pdf_file)
            # e.g., physics.pdf → class11_physics.txt
            txt_file_name = f"{class_folder}_{pdf_file.replace('.pdf', '')}.txt"
            txt_path = os.path.join(PROCESSED_DIR, txt_file_name)

            print(f"📖 Processing {pdf_file} ...")

            try:
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_path)
                
                # Normalize Gujarati text
                text = normalize_gujarati_text(text)
                
                # Check if we got meaningful content
                if len(text.strip()) < 1000:  # If less than 1000 characters
                    print(f"⚠️  Warning: Very little text extracted from {pdf_file}")
                    print("   This might be a scanned PDF without OCR text layer")
                
                # Save extracted text
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"✅ Extracted: {pdf_file} → {txt_file_name}")
                print(f"   Extracted {len(text)} characters")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                import traceback
                traceback.print_exc()

print("🎉 All PDFs processed!")