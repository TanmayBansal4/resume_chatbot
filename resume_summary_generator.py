import os
import docx2txt
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOllama(model="deepseek-llm:7b")

def extract_text_from_file(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        return ""

def summarize_resume(text, folder_name):
    system_prompt = """
You are a resume summarizer. Extract the following from a resume:
- Full Name
- Location
- Contact Info (email, phone number)
- Education (degree + college)
- Work Experience (companies and roles)
- Skills (technical & soft skills)
- Status (Shortlisted or Rejected based on folder name)

Return the response in readable bullet points or paragraphs. Do not return JSON or markdown. Just plain, clean text.
"""

    human_prompt = f"""Resume:
{text}

Folder: {folder_name}
"""

    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=human_prompt.strip())
    ]

    response = model.invoke(messages)
    return response.content.strip()

def append_to_output_file(file_name, summary, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {file_name} =====\n")
        f.write(summary)
        f.write("\n" + "=" * 50 + "\n")

def parse_resumes_to_single_text(folder_path, output_file):
    folder_name = os.path.basename(folder_path)
    all_files = os.listdir(folder_path)

    print(f"\nParsing folder: {folder_path}")
    for file_name in all_files:
        if not file_name.lower().endswith((".pdf", ".docx")):
            continue

        full_path = os.path.join(folder_path, file_name)
        print(f"Processing: {file_name}")

        try:
            text = extract_text_from_file(full_path)
            summary = summarize_resume(text, folder_name)
            name_without_ext = os.path.splitext(file_name)[0]
            append_to_output_file(name_without_ext, summary, output_file)
        except Exception as e:
            print(f"Error with {file_name}: {e}")

shortlisted_folder = "C:/Users/mridu/Downloads/dataset/ShortlistedDS"
rejected_folder = "C:/Users/mridu/Downloads/dataset/RejectedDS"
output_file = "C:/Users/mridu/Downloads/dataset/parsed_resumes.txt"

open(output_file, "w").close()

parse_resumes_to_single_text(shortlisted_folder, output_file)
parse_resumes_to_single_text(rejected_folder, output_file)

print(f"\nAll summaries saved to: {output_file}")
