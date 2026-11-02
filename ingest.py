from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

def parse_front_matter(text):
    metadata = {}
    if not text.startswith("---"):
        return ({},text)
    splits = text.split("---", maxsplit = 2)
    header = splits[1]
    body = splits[2]

    for line in header.strip().splitlines():
        if ": " in line:
            pair = line.split(": ", maxsplit = 1)
            attribute, value = pair[0], pair[1]
            metadata[attribute.strip()] = value.strip()
    return (metadata, body.strip())

def load_plants(folder = "data/plants"):
    docs = []
    files = os.listdir(folder)
    for file in files:
        if not file.endswith(".md"):
            continue
        path = os.path.join(folder,file)
        text = open(path, encoding="utf-8").read()
        metadata, body = parse_front_matter(text)
        header = f"{metadata['name']} (light: {metadata['light']}; toxicity: {metadata['toxicity']})"
        docs.append(Document(page_content=f"{header}\n{body}", metadata=metadata))
    return docs

def build_store():
    docs = load_plants()
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    ids = [doc.metadata['id'] for doc in docs]
    store = Chroma(
        collection_name = "plants",
        embedding_function=embedding_model,
        persist_directory="kb_db"
    )
    store.add_documents(docs, ids = ids)
    return store

def load_store():
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    store = Chroma(
            collection_name = "plants",
            embedding_function=embedding_model,
            persist_directory="kb_db"
        )
    return store

if __name__ == "__main__":
    store = build_store()
    print(f"Built the plant store: {len(store.get()['ids'])} plants in kb_db.")

    opened = load_store()
    print("ok  " if len(opened.get()["ids"]) == 20 else "FAIL", "load_store sees the same 20, without adding")

    snake = opened.get(ids=["snake_plant"])["documents"][0]
    print("ok  " if "toxicity: toxic to cats and dogs" in snake.splitlines()[0] else "FAIL",
          "header line carries the toxicity:", snake.splitlines()[0])
    print("ok  " if "## Feeding" not in snake else "FAIL", "snake plant has no Feeding section (planted trap)")

    for query, expected in [("how often should I water my pothos", "pothos"),
                            ("how often should I water my heartleaf philodendron", "heartleaf_philodendron"),
                            ("is a spider plant safe for cats", "spider_plant")]:
        top = [hit.metadata["id"] for hit in opened.similarity_search(query, k=3)]
        print("ok  " if top[0] == expected else "FAIL", f"{query!r} -> {top}")

    top = [hit.metadata["id"] for hit in opened.similarity_search("how often should I water my orchid", k=3)]
    print("info", f"'orchid' -> {top}   (no orchid sheet: is_relevant must reject these)")