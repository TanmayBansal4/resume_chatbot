import os
import json
import re
from uuid import uuid4

from qdrant_client import QdrantClient, models
from langchain_community.embeddings import OllamaEmbeddings

import google.generativeai as genai

genai.configure(api_key="API_KEY_HERE")
model = genai.GenerativeModel("gemini-2.0-flash")

qdrant = QdrantClient(
    url="URL_HERE",
    api_key="API_KEY_HERE",
)

embedder = OllamaEmbeddings(model="nomic-embed-text")
collection_name = "resume-collection"


def load_chat_history(file="chat_history.json"):
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)
    return []


def save_chat_history(chat_history, file="chat_history.json"):
    with open(file, "w") as f:
        json.dump(chat_history, f, indent=2)


def extract_selected_names(response_text):
    try:
        start = response_text.index("{")
        end = response_text.index("}") + 1
        json_part = response_text[start:end]
        data = json.loads(json_part)
        return data.get("selected_names", [])
    except Exception:
        return []


def find_mentioned_name(user_query, known_names):
    for name in known_names:
        if re.search(rf"\b{name}\b", user_query, re.IGNORECASE):
            return name
    return None


if __name__ == "__main__":
    chat_history = []
    current_entities = {}
    last_used_name = None

    while True:
        user_query = input("➜ You: ").strip()
        if user_query.lower() in {"quit", "exit"}:
            save_chat_history(chat_history)
            break

        chat_history.append({"role": "user", "content": user_query})

        if not current_entities or "someone else" in user_query.lower() or "another" in user_query.lower():
            query_vector = embedder.embed_query(user_query)
            search_result = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=5
            )

            relevant_chunks = "\n\n".join(
                [f"{hit.payload['text'][:300]}" for hit in search_result]
            )

            prompt = (
                "The following is a conversation between a helpful assistant and a user.\n\n"
                f"Relevant Resumes:\n{relevant_chunks}\n\n"
                "Your job is to extract ALL clear candidate names that match the user's query.\n"
                "If you can't find any matching names, reply exactly with: {\"selected_names\": []}\n"
                "DO NOT ask the user for more information.\n"
                "Reply ONLY with valid JSON like this: {\"selected_names\": [\"Name1\", \"Name2\"]}\n"
                f"User Query: {user_query}\n"
                "Your Response:\n"
            )

            response = model.generate_content(prompt).text

            selected_names = extract_selected_names(response)

            if selected_names:
                for name in selected_names:
                    name_vector = embedder.embed_query(name)
                    name_search = qdrant.search(
                        collection_name=collection_name,
                        query_vector=name_vector,
                        limit=1
                    )
                    if name_search:
                        current_entities[name] = name_search[0].id
                        last_used_name = name
            else:
                hit = search_result[0]
                current_entities = { "Unknown": hit.id }
                last_used_name = "Unknown"

        else:
            mentioned = find_mentioned_name(user_query, current_entities.keys())

            if mentioned:
                last_used_name = mentioned

        result = qdrant.retrieve(
            collection_name=collection_name,
            ids=[current_entities[last_used_name]]
        )[0]
        final_payload = result.payload

        prompt = (
            "The following is a conversation between a helpful assistant and a user.\n\n"
        )
        for turn in chat_history[-10:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"

        prompt += f"\nCandidate Info for {last_used_name}:\n{final_payload['text']}\n\nAssistant:"

        bot_response = model.generate_content(prompt).text
        print(f"\n🤖 {bot_response.strip()}\n")

        chat_history.append({"role": "assistant", "content": bot_response.strip()})
        save_chat_history(chat_history)
