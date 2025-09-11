# src/preprocessing/pdf_to_text_all.py

import fitz  # PyMuPDF
import os

# Paths
TEXTBOOK_DIR = "data/textbooks"
PROCESSED_DIR = "data/processed"

# Create processed folder if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

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
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text("text") + "\n"

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"✅ Extracted: {pdf_file} → {txt_file_name}")
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")

print("🎉 All PDFs processed!")
