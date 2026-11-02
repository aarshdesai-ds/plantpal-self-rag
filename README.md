# PlantPal: Self-RAG for Houseplant Care

A LangGraph question-answering workflow built on the Self-RAG pattern. It answers houseplant questions from 20 plant care sheets. It decides whether a question needs the sheets at all, keeps only the sheets about the plant asked about, checks every answer against those sheets before showing it, and says plainly when a sheet doesn't cover something instead of filling the gap with plausible advice.

The project was built in three steps, and the middle one took six measured attempts. Each change was run on a labelled question set, and the answers were read as well as counted. The sections below report every attempt, including the ones that made things worse.

*This is an independent learning project, not plant care advice. The care sheets and questions are synthetic and were written for this project.*

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [The Care Sheets](#the-care-sheets)
- [Step 1: The Baseline](#step-1-the-baseline)
- [Step 2: Relevance, Support and a Safety Net](#step-2-relevance-support-and-a-safety-net)
- [Step 3: Usefulness and Rewriting](#step-3-usefulness-and-rewriting)
- [Results](#results)
- [How It Differs from the Course Notebook](#how-it-differs-from-the-course-notebook)
- [What the Project Taught](#what-the-project-taught)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Known Limitations](#known-limitations)
- [Ideas for Extension](#ideas-for-extension)

---

## What It Does

```
Q: How often should I feed my snake plant, and is it safe for cats?

Baseline:  Feed your snake plant every 4 to 6 weeks during the growing season (spring and summer)
           with a balanced fertilizer. It is toxic to cats if eaten [snake_plant].
           ← the snake plant sheet has no feeding section; the frequency was invented

PlantPal:  It is toxic to cats if eaten [snake_plant].
           The care sheet doesn't cover feeding frequency or fertilizer type.
```

```
Q: Is English ivy toxic to dogs, and what symptoms does it cause?

Baseline:  Yes... Ingestion can lead to symptoms such as vomiting, abdominal pain, diarrhea...
           ← answered from memory; the sheet only says "Toxic to pets."

PlantPal:  Yes, English ivy is toxic to dogs [english_ivy].
           The care sheet doesn't cover symptoms of toxicity.
```

```
Q: How often should I water my orchid?

Baseline:  Water your orchid when the top 2 to 3 cm of soil is dry. [parlor_palm]
           ← there is no orchid sheet; this came from the parlor palm

PlantPal:  The care sheets don't cover that.
```

---

## How It Works

```
START ──► decide_retrieval ──┬── general question ──► answer_direct ──► END
                             └── plant question ──► retrieve ◄──────────────────────────┐
                                                       │                                │
                                                       ▼                                │
                                                  is_relevant ──┬── nothing relevant, first try ──► rewrite_query ──┘
                                                                ├── nothing relevant after a rewrite ──► no_answer ──► END
                                                                └── relevant sheets ──► generate
                                                                                           │
                                                                                           ▼
                                                                          ┌──────────► is_sup ──┬── fully supported ──► is_use ──► END
                                                                          │                     ├── first miss ──► revise ──┐
                                                                          └─────────────────────────────────────────────────┘
                                                                                                └── still unsupported ──► fallback ──► END
```

| Node | Does | Decided by |
|---|---|---|
| `decide_retrieval` | Skips the sheets for general questions. A question naming a specific plant always retrieves | `gpt-4o-mini`, structured output |
| `retrieve` | The 4 nearest sheets from Chroma, using the rewritten query if there is one | Code |
| `is_relevant` | Keeps a sheet only if it is about the plant asked about, one call per sheet | Model, structured output |
| `generate` | Answers from the kept sheets, citing each plant id | Model |
| `is_sup` | Fully, partially or not supported, with evidence quotes and a list of unsupported claims | Model, then **code**: every evidence quote must appear in the sheets, and "the sheet doesn't cover X" statements are not counted as claims |
| `revise` | Deletes each sentence with an unsupported claim and says plainly that the sheet doesn't cover that point. At most once | Model, capped in code |
| `fallback` | If the answer is still unsupported after its revision, it is replaced by the verified evidence quotes, or by "The care sheets don't cover that." | **Code only**, no model call |
| `is_use` | Does the answer address the question? | Model, structured output |
| `rewrite_query` | A better search query with the plant's other common names. At most once, and only when no relevant sheet was found | Model writes, code decides when |
| LangSmith | Every run is traced with `run_name`, `tags` and `metadata` (eval row id and group) | — |

---

## The Care Sheets

There are 20 synthetic plant care sheets, one Markdown file each, with front matter and six sections (Light, Water, Soil, Feeding, Pets, Common problems):

```
id: snake_plant
name: Snake Plant
light: low to bright indirect
toxicity: toxic to cats and dogs
```

`ingest.py` stores each sheet whole, with a header line at the top of its text ("Snake Plant (light: low to bright indirect; toxicity: toxic to cats and dogs)"), so the toxicity is in the text the model reads, not only in the metadata.

**Planted traps:**

| Trap | Plants | Tests |
|---|---|---|
| No Feeding section | snake plant, ZZ plant, jade plant, string of pearls, Christmas cactus | Does the answer invent feeding advice? |
| Vague toxicity: "Toxic to pets." with no details | English ivy, dracaena | Does the answer invent symptoms? |
| Look-alikes with different facts | pothos and heartleaf philodendron, snake plant and ZZ plant | Does relevance pick the right sheet? |
| Not in the library | orchid, Venus flytrap, bonsai | Does it say so instead of answering from a neighbour's sheet? |

### The question set

`eval/questions.jsonl` has 20 labelled questions:

| Group | Rows | Expected |
|---|---|---|
| general | 4 | No retrieval, a direct answer |
| answerable | 6 | The right sheet, fully supported |
| partly missing | 5 | Answer the covered part, and say the rest isn't covered. Four ask about feeding a plant whose sheet has none |
| not in library | 3 | "The care sheets don't cover that." |
| vague | 2 | No plant named. Read by hand |

Each run is scored in code (retrieval decisions, the right sheet cited, "not covered" on the missing plants, and a regex for invented feeding frequencies) and then read.

---

## Step 1: The Baseline

Decide whether to retrieve, retrieve 4 sheets, answer with citations. No relevance check and no support check.

| Measure | Result |
|---|---|
| Retrieval decisions | 19 of 20. P05 ("Is English ivy toxic to dogs?") skipped the sheets and answered from memory |
| Invented feeding frequency | 3 of 4: "every 4 to 6 weeks", "every 6 weeks", "every 4-6 weeks" for plants whose sheets have no feeding section |
| Wrong sheet cited | 3: the orchid answered from the parlor palm, ZZ plant questions answered from the snake plant and the parlor palm |
| Not-in-library questions answered anyway | 3 of 3 |

The first run of the frequency regex reported 2 of 4, because it expected spaces around the dash in "4-6". The pattern was fixed and the count corrected.

---

## Step 2: Relevance, Support and a Safety Net

### Relevance first

One rule was added to `decide_retrieval` ("if the question names a specific plant, always retrieve"), and `is_relevant` was added before generation.

| Measure | Baseline | With relevance |
|---|---|---|
| Retrieval decisions | 19/20 | 20/20 |
| Not-in-library → "not covered" | 0/3 | 3/3 |
| Wrong sheet cited | 3 | 0 |
| Invented feeding frequency | 3/4 | **3/4** |

Relevance fixed every problem caused by the wrong sheet and none caused by the right one. Read by hand, all five partly missing questions still invented something, and P05 got worse: the invented symptoms were now cited to the ivy sheet.

### Then the support check, in six attempts

| Attempt | Change | Invented frequency | Fell back | What reading the answers showed |
|---|---|---|---|---|
| 1 | IsSUP and a revise step, with an example sentence in the revise prompt | 0/4 | — | The example, "The care sheet doesn't say how often to feed it.", was copied into answers about ivy toxicity and yellow leaves |
| 2 | The example removed | **3/4** | — | Revise now **added** a "not covered" sentence next to the invented claim instead of replacing it |
| 3 | A fallback in code: an answer still unsupported after its revision is replaced by the verified evidence quotes | 0/4 | 4 | Safe, but revise fixed none of the four. The fallback did all the work |
| 4 | Revise rewritten as two explicit steps: delete the sentence, then say it isn't covered | 0/4 | 4 | Revise now worked, but the checker flagged its honest "doesn't cover" sentences as unsupported claims |
| 5 | In code, "doesn't cover / not covered" statements are not counted as claims | 0/4 | 3 | P05 passed with a readable answer, produced by revise |
| 6 | In code, a bare topic like "feeding frequency" is not a claim when the answer itself says it isn't covered | 0/4 | **1** | Three readable honest answers from revise. The one fallback (P02) had kept its invented sentence, correctly caught |

Two problems showed up during these runs that were not about the support check:

- **The example in the prompt was doing two jobs.** It leaked into other answers, but it was also showing the model that a "not covered" sentence replaces a claim. Removing it fixed the leak and broke the replacement.
- **A search index can go stale.** In attempt 6, the monstera question stopped retrieving the monstera sheet, although the store held all 20 sheets and the monstera text and embedding were intact. After deleting `kb_db` and building it once from scratch, it came back. Re-running the ingest over an existing store is the likely cause.

---

## Step 3: Usefulness and Rewriting

`is_use` judges whether a supported answer addresses the question. `rewrite_query` writes a better search query and retries once.

| Run | Rewrites | Answers improved | What reading the answers showed |
|---|---|---|---|
| Rewrite whenever an answer is not useful, or nothing relevant is found | 5 | 0 | The three missing plants were rewritten and still not covered. P01's honest partial answer was judged not useful and retried for nothing. **P04 got worse:** after the retry it was padded with "Mushy pearls mean overwatering…", which nobody asked about |
| Rewrite only when no relevant sheet was found, decided in code | 3 | 0 | No more retries on answers that already had the right sheet, and no padding |

The rewrites themselves were good: they added real alternative names ("snake plant, mother-in-law's tongue", "string of pearls, Senecio rowleyanus"). But the question set had no real retrieval miss left once the index was rebuilt, so the loop never had something to find. **A rewrite can fix retrieval. It can't fix knowledge that isn't in the sheets,** and judging an honest "not covered" as unhelpful pushes the model towards filler.

---

## Results

| Measure | Baseline | Final |
|---|---|---|
| Retrieval decisions | 19/20 | 20/20 |
| Wrong sheet cited | 3 | 0 |
| Not-in-library questions answered from memory or a neighbour's sheet | 3/3 | 0/3 |
| Invented feeding frequency | 3/4 | 0/4 |
| Invented content of any kind on the 5 partly missing questions, by reading | 5/5 (with relevance only) | 1/5 |
| Answers that still contained unsupported claims after revision, reaching the user | — | 0: the fallback replaces them |

20 synthetic questions, `gpt-4o-mini` at temperature 0, one run per change. Revise and the judges vary between runs, so changes of one row are within normal variation.

---

## How It Differs from the Course Notebook

| Notebook | PlantPal | Why |
|---|---|---|
| Up to 10 revisions and 3 rewrites, with a recursion limit of 80 | At most 1 of each, enforced in code | Each extra loop costs calls and rarely changes the answer |
| The judge's evidence quotes are trusted | Every quote must appear in the sheets, checked in code | The judge can invent its evidence too |
| After the maximum retries, the answer is accepted | An answer still unsupported is replaced by verified quotes | The check is useless if its verdict isn't acted on |
| A quote-only reviser | A readable reviser, with quote-only as the code fallback | Supported but unreadable answers aren't much use |
| Relevance at the topic level ("same topic area counts") | Relevance at the plant level | A jade plant sheet is about succulents, but it isn't about the snake plant |
| Rewrite whenever the answer isn't useful | Rewrite only when no relevant sheet was found | Rewriting can't add facts the right sheet doesn't have |

---

## What the Project Taught

- **The right document can still produce a wrong answer.** Relevance fixed every wrong-sheet problem and none of the inventions.
- **An example in a prompt is an instruction.** It leaked into unrelated answers, and removing it changed behaviour nobody had noticed it was controlling.
- **Act on the check, or it isn't a check.** Until the fallback existed, an answer that failed the support check was shown anyway.
- **Decide in code what the model keeps getting wrong.** The judge ignored its own rule about "not covered" sentences in every run; a regex fixed it in every run.
- **Measure every change, including the reverts.** Removing the example looked like a pure fix and took invented frequencies from 0/4 back to 3/4.
- **Usefulness and honesty pull against each other.** A retry aimed at more useful answers produced filler.
- **Stale indexes look like model problems.** A missing monstera answer turned out to be the vector store.

---

## Project Structure

```
plantpal/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: OPENAI_API_KEY (and LangSmith keys, if used)
│
├── data/
│   └── plants/             # 20 synthetic care sheets with front matter
│
├── ingest.py               # Care sheets → Chroma (kb_db/), one document per sheet, toxicity in the text
├── self_rag_graph.py       # State, prompts, all nodes, routers and the graph, with free checks and the step3 run
│
└── eval/
    └── questions.jsonl     # 20 labelled questions (general, answerable, partly missing, not in library, vague)
```

`kb_db/` is created when the project runs, and is not committed.

---

## How to Run

1. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

2. Create a `.env` file with your `OPENAI_API_KEY`.

3. Build the care sheet store (one embedding call). To rebuild it, delete `kb_db` first rather than running the ingest over an existing store:

```bash
python ingest.py
```

4. Run the free checks (39 checks, no model calls):

```bash
python self_rag_graph.py
```

5. Run all 20 questions through the full graph (about 150 `gpt-4o-mini` calls):

```bash
python self_rag_graph.py step3
```

---

## Known Limitations

- **Invented product types get through.** "A balanced fertilizer formulated for succulents" for the jade plant passed the support check in every run. The checker catches invented frequencies far more reliably than invented product types.
- **Claims merged across sheets get through.** A vague question about yellow leaves got "let the soil dry out completely… for both the ZZ plant and pothos", but the pothos sheet says the top 5 cm.
- **The usefulness judge is too strict on honest partial answers.** P01 is flagged not useful even though it answers the toxicity half and says the feeding half isn't covered. It is only a label now; it no longer triggers a retry.
- **Revise is inconsistent.** On the same question, it sometimes removes an invented claim and sometimes keeps it. The fallback makes this safe, but a fallback reply of verified quotes is less readable.
- **The fallback can be noisy.** It quotes all verified evidence, which may include facts not asked about, or the sheet's header line.
- **Retrying a missing plant costs calls.** An orchid question is rewritten once before ending "not covered".
- **The search index can go stale** if `ingest.py` is run over an existing store. Delete `kb_db` before rebuilding.
- **The evaluation is small and synthetic.** 20 questions written by the same hand as the sheets, one model, one run per change.

---

## Ideas for Extension

1. **Alias questions** ("Swiss cheese plant", "mother-in-law's tongue") to test the rewrite on a real retrieval miss.
2. **A product-type check in code,** like the frequency regex, for claims the judge lets through.
3. **Per-sentence support checking,** so a claim merged across two sheets is checked against each one.
4. **An evaluation file per run,** saving every answer, so runs can be compared row by row and re-scored for free.
5. **Ask which plant is meant** when a question names none, instead of answering for several.
