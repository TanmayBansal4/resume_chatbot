import os
import json
from uuid import uuid4

from qdrant_client import QdrantClient, models
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


llm = Ollama(model="deepseek-llm:7b")
embedder = OllamaEmbeddings(model="nomic-embed-text")

qdrant = QdrantClient(
    url="url to qdrant or localhost",
    api_key="API_Key here",
)

collection_name = "resume-collection"



def load_resume_summaries_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shortlisted, rejected = [], []

    for entry in data:
        summary = entry.get("Summary", "").strip()
        status = entry.get("Folder", "").lower()
        if "shortlist" in status:
            shortlisted.append({"text": summary, "status": "Shortlisted"})
        elif "reject" in status:
            rejected.append({"text": summary, "status": "Rejected"})
        else:
            print(f"⚠️ Unknown status for entry: {status}")

    return shortlisted, rejected


def save_chat_history(chat_history, file="chat_history.json"):
    with open(file, "w") as f:
        json.dump(chat_history, f, indent=2)


def load_chat_history(file="chat_history.json"):
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    summaries_file = "PATH_TO/parsed_resumes.json"

    print("📄 Reading labeled resume summaries from JSON...")
    shortlisted, rejected = load_resume_summaries_from_json(summaries_file)
    print(f"✅ Loaded: {len(shortlisted)} shortlisted, {len(rejected)} rejected")

    all_resumes = shortlisted + rejected
    texts = [entry["text"] for entry in all_resumes]

    print("🔢 Embedding resumes...")
    vectors = embedder.embed_documents(texts)
    ids = [str(uuid4()) for _ in texts]

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
                payload={
                    "text": texts[i],
                    "status": all_resumes[i]["status"]
                }
            )
            for i in range(len(all_resumes))
        ]
    )
    print("✅ All resumes uploaded to Qdrant!")

    print("\n🤖 Ready to chat! Memory is ON. Type 'quit' to exit.\n")

    chat_history = load_chat_history()

    while True:
        user_query = input("➜ You: ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("👋 Exiting. Your chat history is saved.")
            save_chat_history(chat_history)
            break

        chat_history.append({"role": "user", "content": user_query})

        query_vector = embedder.embed_query(user_query)
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )

        relevant_chunks = "\n\n".join([hit.payload["text"][:300] for hit in search_result])

        prompt = "The following is a conversation between a helpful assistant and a user.\n\n"
        for turn in chat_history[-10:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"

        prompt += f"\nRelevant Resume Chunks:\n{relevant_chunks}\n\n"
        prompt += f"Assistant:"

        bot_response = llm(prompt)
        print(f"\n🤖 {bot_response.strip()}\n")

        chat_history.append({"role": "assistant", "content": bot_response.strip()})

        save_chat_history(chat_history)
