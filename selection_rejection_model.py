import os
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

def load_resume_summaries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_summaries(text):
    shortlisted = []
    rejected = []
    current = []
    label = None

    for line in text.splitlines():
        if "Shortlisted" in line:
            if current and label:
                (shortlisted if label == "Shortlisted" else rejected).append("\n".join(current).strip())
            current = []
            label = "Shortlisted"
        elif "Rejected" in line:
            if current and label:
                (shortlisted if label == "Shortlisted" else rejected).append("\n".join(current).strip())
            current = []
            label = "Rejected"
        else:
            current.append(line)
    
    if current and label:
        (shortlisted if label == "Shortlisted" else rejected).append("\n".join(current).strip())

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
    summaries_file = "path_to/parsed_resumes.txt"
    new_resume_folder = "path_to/new_resumes"

    print("Reading labeled resume summaries...")
    summaries = load_resume_summaries(summaries_file)

    print("Separating shortlisted and rejected summaries...")
    shortlisted, rejected = split_summaries(summaries)

    print("Encoding vectors...")
    shortlisted_embeds = build_embeddings(shortlisted)
    rejected_embeds = build_embeddings(rejected)

    print("\nScreening new resumes by similarity...\n")
    process_new_resumes(new_resume_folder, shortlisted_embeds, rejected_embeds)

    print("\nResume similarity screening complete.")
