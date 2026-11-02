from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from ingest import load_store
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
import re

load_dotenv()

store = load_store()
retriever = store.as_retriever(search_kwargs = {"k":4})
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

FEED_WORDS = re.compile(r"\b(feed|feeding|fed|fertili[sz](e|er|ing))\b", re.IGNORECASE)
FREQUENCY = re.compile(
    r"\b(daily|weekly|fortnightly|monthly|yearly|annually"
    r"|every\s+(\d+\s*(to|-|-)\s*)?\d+\s+(day|week|month)s?"
    r"|every\s+(other\s+|few\s+)?(day|week|month)s?"
    r"|(once|twice)\s+(a|per|every)\s+(day|week|month|year)"
    r"|\d+\s+((to|-)\s+\d+\s+)?times\s+(a|per)\s+(week|month|year))\b",
    re.IGNORECASE)

def invents_frequency(answer):
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    return any(FEED_WORDS.search(s) and FREQUENCY.search(s) for s in sentences)

def format_docs(docs):
    blocks = []
    for doc in docs:
        block = f"[{doc.metadata['id']}]\n{doc.page_content}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

class GraphState(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    answer: str

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(... , description="True if answering needs care facts about a specific plant from the care sheets")

decide_prompt = ChatPromptTemplate([
    ("system", """You decide whether a houseplant question needs the plant care sheets.
    - Retrieve if it asks about a specific plant's care: watering, light, soil, feeding, pets, problems.
    - Don't retrieve for general plant knowledge: definitions, or how something works in general.
    - If unsure, retrieve."""),
    ("human", "Question: {question}")
])

decide_chain = decide_prompt | llm.with_structured_output(RetrieveDecision)

direct_prompt = ChatPromptTemplate([
    ("system", "You are a houseplant assistant. Answer general plant questions briefly from general knowledge."),
    ("human", "{question}")
])

direct_chain = direct_prompt | llm

answer_prompt = ChatPromptTemplate([
    ("system", "You are a houseplant assistant. Answer the question using the care sheets below. Cite the plant id you used in square brackets, like [pothos]. Keep it short."),
    ("human", "Question: {question}\n\nCare sheets:\n{context}")
])

answer_chain = answer_prompt | llm

def decide_retrieval(state: GraphState):
    need_retrieval = decide_chain.invoke({"question": state['question']}).should_retrieve
    return {"need_retrieval": need_retrieval}

def answer_direct(state: GraphState):
    answer = direct_chain.invoke({"question": state['question']}).content
    return {"answer": answer}

def retrieve(state: GraphState):
    docs = retriever.invoke(state['question'])
    return {"docs": docs}

def generate(state: GraphState):
    context = format_docs(state['docs'])
    question = state['question']
    answer = answer_chain.invoke({"question": question, "context": context}).content
    return {"answer": answer}

def route_after_decide(state: GraphState) -> Literal['retrieve', 'answer_direct']:
    if state['need_retrieval']:
        return "retrieve"
    return "answer_direct"

graph = StateGraph(GraphState)

graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("retrieve", retrieve)
graph.add_node("answer_direct", answer_direct)
graph.add_node("generate", generate)

graph.add_edge(START, "decide_retrieval")

graph.add_conditional_edges("decide_retrieval", route_after_decide,{
    "retrieve": "retrieve",
    "answer_direct": "answer_direct"
})

graph.add_edge("retrieve", "generate")
graph.add_edge("answer_direct", END)
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    import sys
    import json

    print("--- graph (free) ---")
    print(sorted(app.get_graph().nodes))
    sample = format_docs(retriever.invoke("how often should I water my pothos"))
    print("ok  " if "[pothos]" in sample else "FAIL", "sheets are labelled with their ids")

    print("\n--- invents_frequency (free) ---")
    for text, expected in [("Feed it monthly in spring.", True), ("Fertilise every 4 weeks.", True),
                           ("Feed it once a month.", True), ("Feed 2 to 3 times a year.", True),
                           ("The sheet doesn't say how often to feed it.", False), ("It is toxic to cats.", False)]:
        print("ok  " if invents_frequency(text) == expected else "FAIL", f"{text!r} -> {invents_frequency(text)}")

    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        print("\n--- baseline on all 20 questions (about 20-36 model calls) ---")
        rows = [json.loads(line) for line in open("eval/questions.jsonl", encoding="utf-8") if line.strip()]
        decisions = invented = 0
        for row in rows:
            result = app.invoke({"question": row["question"]},
                                config={"run_name": "plantpal", "tags": ["plantpal", "test", "baseline"],
                                        "metadata": {"row_id": row["id"], "group": row["group"]}})
            decided = result["need_retrieval"]
            decisions += decided == row["expected_retrieve"]
            flag = ""
            if row["missing_part"] == "feeding" and invents_frequency(result["answer"]):
                invented += 1
                flag = "   <-- INVENTED FEEDING"
            retrieved = [doc.metadata["id"] for doc in result.get("docs", [])]
            print(f"\n{row['id']} {row['group']:15} retrieve={decided} (expected {row['expected_retrieve']}) {retrieved}{flag}")
            print(f"   A: {result['answer'][:220]}")
        print(f"\nretrieval decisions: {decisions}/20   invented feeding advice: {invented}/4 (P01-P04)")