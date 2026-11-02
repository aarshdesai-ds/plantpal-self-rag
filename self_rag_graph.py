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
    r"|every\s+(\d+\s*(to|-|–)\s*)?\d+\s+(day|week|month)s?"
    r"|every\s+(other\s+|few\s+)?(day|week|month)s?"
    r"|(once|twice)\s+(a|per|every)\s+(day|week|month|year)"
    r"|\d+\s*((to|-|–)\s*\d+\s*)?times\s+(a|per)\s+(week|month|year))\b",
    re.IGNORECASE)

NOT_COVERED = "The care sheets don't cover that."



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
    relevant_docs: List[Document]
    context: str
    issup: Literal['fully_supported', "partially_supported", "no_support"]
    evidence: list[str]


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(... , description="True if answering needs care facts about a specific plant from the care sheets")

decide_prompt = ChatPromptTemplate([
    ("system", """You decide whether a houseplant question needs the plant care sheets.
    - Retrieve if it asks about a specific plant's care: watering, light, soil, feeding, pets, problems.
    - Don't retrieve for general plant knowledge: definitions, or how something works in general.
    - If the question names a specific plant, always retrieve, even if you think you know the answer.
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

class Relevance(BaseModel):
    is_relevant: bool = Field(... , description = "True only if this care sheet is about the plant the question asks about. False for a different plant, even if it covers the same topic")
    
relevance_prompt = ChatPromptTemplate([
    ("system", "You judge whether ONE plant care sheet is relevant to a question. It is relevant only if it is about the plant the question asks about. A sheet about a different plant is NOT relevant, even if it covers the same topic. If the question names no plant, it is relevant only if it directly answers the question as asked."),
    ("human","Question: {question}\n\nCare sheet:\n{sheet}")
])
relevance_model = llm.with_structured_output(Relevance)
relevance_chain = relevance_prompt | relevance_model


def decide_retrieval(state: GraphState):
    need_retrieval = decide_chain.invoke({"question": state['question']}).should_retrieve
    return {"need_retrieval": need_retrieval}

def route_after_decide(state: GraphState) -> Literal['retrieve', 'answer_direct']:
    if state['need_retrieval']:
        return "retrieve"
    return "answer_direct"

def answer_direct(state: GraphState):
    answer = direct_chain.invoke({"question": state['question']}).content
    return {"answer": answer}

def retrieve(state: GraphState):
    docs = retriever.invoke(state['question'])
    return {"docs": docs}

def generate(state: GraphState):
    context = format_docs(state['relevant_docs'])
    question = state['question']
    answer = answer_chain.invoke({"question": question, "context": context}).content
    return {"answer": answer, "context":context}

def no_answer(state: GraphState):
    return {"answer": NOT_COVERED}

def is_relevant(state: GraphState):
    relevant_docs = []
    question = state['question']
    for doc in state['docs']:
        relevant = relevance_chain.invoke({"question": question, "sheet": doc.page_content}).is_relevant
        if relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: GraphState) -> Literal['no_answer', 'generate']:
    if state.get("relevant_docs",[]):
        return "generate"
    return "no_answer"


graph = StateGraph(GraphState)

graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("retrieve", retrieve)
graph.add_node("answer_direct", answer_direct)
graph.add_node("generate", generate)
graph.add_node("is_relevant", is_relevant)
graph.add_node("no_answer", no_answer)

graph.add_edge(START, "decide_retrieval")

graph.add_conditional_edges("decide_retrieval", route_after_decide,{
    "retrieve": "retrieve",
    "answer_direct": "answer_direct"
})

graph.add_edge("retrieve", "is_relevant")

graph.add_conditional_edges("is_relevant", route_after_relevance, {
    "no_answer": "no_answer",
    "generate": "generate"
})

graph.add_edge("answer_direct", END)
graph.add_edge("no_answer", END)
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
                           ("Feed every 4-6 weeks.", True), ("Feed every 4–6 weeks.", True),
                           ("Feed 2-3 times a year.", True),
                           ("The sheet doesn't say how often to feed it.", False), ("It is toxic to cats.", False)]:
        print("ok  " if invents_frequency(text) == expected else "FAIL", f"{text!r} -> {invents_frequency(text)}")

    print("\n--- routers (free) ---")
    print("ok  " if route_after_relevance({"relevant_docs": []}) == "no_answer" else "FAIL", "nothing relevant -> no_answer")
    print("ok  " if route_after_relevance({"relevant_docs": [Document(page_content="x", metadata={"id": "pothos"})]}) == "generate"
          else "FAIL", "something relevant -> generate")

    if len(sys.argv) > 1 and sys.argv[1] == "step2a":
        print("\n--- step 2a: is_relevant added, on all 20 questions (about 100 model calls) ---")
        citation = re.compile(r"\[([a-z_]+)\]")
        rows = [json.loads(line) for line in open("eval/questions.jsonl", encoding="utf-8") if line.strip()]
        decisions = invented = not_covered_ok = answered_ok = wrong_cites = 0
        for row in rows:
            result = app.invoke({"question": row["question"]},
                                config={"run_name": "plantpal", "tags": ["plantpal", "test", "step2a"],
                                        "metadata": {"row_id": row["id"], "group": row["group"]}})
            decisions += result["need_retrieval"] == row["expected_retrieve"]
            retrieved = [doc.metadata["id"] for doc in result.get("docs", [])]
            relevant = [doc.metadata["id"] for doc in result.get("relevant_docs", [])]
            cited = citation.findall(result["answer"])
            flags = []
            if row["missing_part"] == "feeding" and invents_frequency(result["answer"]):
                invented += 1
                flags.append("INVENTED FEEDING")
            if any(c not in relevant for c in cited):
                wrong_cites += 1
                flags.append("CITES A SHEET NOT KEPT")
            if row["group"] == "not_in_library":
                not_covered_ok += result["answer"] == NOT_COVERED
            if row["group"] == "answerable":
                answered_ok += row["plant_id"] in cited
            print(f"\n{row['id']} {row['group']:15} retrieve={result['need_retrieval']} retrieved={retrieved}")
            print(f"   relevant: {relevant}" + (f"   <-- {', '.join(flags)}" if flags else ""))
            print(f"   A: {result['answer'][:220]}")
        print(f"\nretrieval decisions: {decisions}/20 (baseline 19/20)")
        print(f"invented feeding advice: {invented}/4 (baseline 3/4)")
        print(f"not-in-library answered 'not covered': {not_covered_ok}/3 (baseline 0/3)")
        print(f"answerable rows citing the right sheet: {answered_ok}/6 (baseline 5/6, A03 cited snake_plant)")
        print(f"answers citing a sheet the filter didn't keep: {wrong_cites}")