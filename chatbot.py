import os
import re
from uuid import uuid4

from qdrant_client import QdrantClient, models
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

llm = Ollama(model="deepseek-r1:7b")
embedder = OllamaEmbeddings(model="nomic-embed-text")

qdrant = qdrant_client = QdrantClient(
    url="qdrant_url_here",
    api_key="api_key_here",
)


collection_name = "resume-collection"

def load_resume_summaries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_summaries(text):
    shortlisted = []
    rejected = []

    blocks = re.split(r"={5,}", text)  # split on =====

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        match = re.search(r"\*\*Status:\*\*\s*(Shortlisted|Rejected)", block, re.IGNORECASE)
        if match:
            status = match.group(1).strip().lower()
            if status.startswith("shortlist"):
                shortlisted.append(block)
            elif status.startswith("reject"):
                rejected.append(block)

    return shortlisted, rejected

if __name__ == "__main__":
    summaries_file = "path_to/parsed_resumes.txt"

    print("📄 Reading labeled resume summaries...")
    summaries = load_resume_summaries(summaries_file)

    print("🔀 Splitting summaries...")
    shortlisted, rejected = split_summaries(summaries)

    print(f"✅ Loaded: {len(shortlisted)} shortlisted, {len(rejected)} rejected")

    print("🔢 Embedding resumes...")
    all_resumes = shortlisted + rejected
    vectors = embedder.embed_documents(all_resumes)
    ids = [str(uuid4()) for _ in all_resumes]

    if not qdrant.collection_exists(collection_name=collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE
            )
        )
        print(f"✅ Created collection: {collection_name}")

    qdrant.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload={"text": all_resumes[i]}
            )
            for i in range(len(all_resumes))
        ]
    )

    print("✅ All resumes uploaded to Qdrant!")

    print("\n🤖 Let’s find the best candidates for your requirement!")

    while True:
        user_query = input("What skills, experience or criteria do you want? (type 'quit' to exit) ➜ ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("👋 Exiting. Have a great day!")
            break

        query_vector = embedder.embed_query(user_query)

        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )

        print("\n🔍 Top Matches:\n")
        for i, hit in enumerate(search_result, 1):
            payload = hit.payload
            score = hit.score
            print(f"#{i} — Score: {score:.4f}\n")
            print(payload["text"][:500])  # Show first 500 chars
            print("\n" + "=" * 40 + "\n")
