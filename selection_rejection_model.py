import os
import json
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_resume_summaries_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shortlisted = []
    rejected = []

    for entry in data:
        summary = entry.get("Summary", "").strip()
        folder = entry.get("Folder", "").lower()
        status = entry.get("Folder", "").lower()

        if "shortlist" in status:
            shortlisted.append(summary)
        elif "reject" in status:
            rejected.append(summary)

    return shortlisted, rejected

def build_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=True)

def decide_by_similarity(new_resume, shortlisted_embeds, rejected_embeds):
    new_embed = embedder.encode(new_resume, convert_to_tensor=True)

    short_sim = util.cos_sim(new_embed, shortlisted_embeds).max().item()
    reject_sim = util.cos_sim(new_embed, rejected_embeds).max().item()

    print(f"Similarity: Shortlisted={short_sim:.3f} | Rejected={reject_sim:.3f}")

    if reject_sim > short_sim:
        return "REJECTED"
    else:
        return "SHORTLISTED"

def process_new_resumes(folder_path, shortlisted_embeds, rejected_embeds):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        resume_text = ""

        try:
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    resume_text = f.read().strip()
            elif filename.endswith(".docx"):
                resume_text = extract_text_from_docx(file_path)
            elif filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_path)

            if not resume_text:
                print(f"\n--- {filename} ---\n[Empty or unreadable resume file]\n")
                continue

            print(f"\nProcessing {filename}...")
            decision = decide_by_similarity(resume_text, shortlisted_embeds, rejected_embeds)
            print(f"\n--- {filename} ---\nDecision: {decision}\n")

        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    json_file = "PATH_TO/parsed_resumes.json"
    new_resume_folder = "PATH_TO/new_resumes"

    print("Reading labeled resume summaries from JSON...")
    shortlisted, rejected = load_resume_summaries_from_json(json_file)

    print(f"Loaded {len(shortlisted)} shortlisted and {len(rejected)} rejected summaries.")

    print("Encoding vectors...")
    shortlisted_embeds = build_embeddings(shortlisted)
    rejected_embeds = build_embeddings(rejected)

    print("\nScreening new resumes by similarity...\n")
    process_new_resumes(new_resume_folder, shortlisted_embeds, rejected_embeds)

    print("\nResume similarity screening complete.")
