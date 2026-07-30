# Generating and reporting results

This is the playbook for producing the measured results this repository
currently lacks. The README's Evaluation section honestly states that no
systematic evaluation has been run; following this document replaces that
absence with real numbers and real examples, without ever fabricating a
claim. Everything here after the index build runs from the repository
root with the project virtualenv active.

The guiding rule: **only report numbers that a command in this document
actually produced.** Structural metrics (distribution, duplicates,
coverage) are evidence of pipeline health, not of question quality; keep
the two claims separate when writing any of this up.

## 0. Prerequisites

1. Python 3.13+, dependencies installed:

   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. Credentials. Copy the template and fill in the three required values
   (an Azure OpenAI key plus chat and embedding endpoints; deployment
   names default to `gpt-4o` and `text-embedding-3-small`):

   ```bash
   cp .env.example .env
   ```

3. A folder of the course reading PDFs, e.g. `./course_readings/`.
   Published papers and lecture notes only -- the corpus must contain no
   patient data, since the README states it never has.

4. Confirm the suite is green before spending API budget:

   ```bash
   python -m pytest -q
   ```

## 1. Build the vector store (once)

```bash
python build_RAG.py --pdf-folder ./course_readings --vec-store ./vec_store
```

Embedding cost is negligible next to generation (text-embedding-3-small
prices per million tokens; a full course corpus is typically a few
million tokens at most). Then smoke-test retrieval before generating
anything:

```bash
python build_RAG.py --pdf-folder ./course_readings --vec-store ./vec_store --smoke-test-query "What are the main themes of artificial intelligence in medicine?"
```

If the response reads like it came from the course readings, retrieval
works. If it is empty or generic, fix this before continuing -- every
downstream number depends on it.

## 2. Generate a baseline week

Pick a week whose readings you know well (you will be judging the output
by eye first). Twelve questions is enough to be interesting and cheap
enough to regenerate:

```bash
python generate_quiz.py --week-number 3 --num-questions 12 --vec-store ./vec_store --output-dir ./quizzes --seed 0
```

Each question costs two model calls (one retrieval-and-summarize, one
authoring). Watch the closing line: `Generated 12/12 questions (0
responses failed to parse)` is itself a reportable reliability datum --
record it. Repeat for a second week (e.g. `--week-number 7`) so no
number in the README rests on a single week.

## 3. Generate a refined week (the round-table showcase)

```bash
python generate_quiz.py --week-number 3 --num-questions 6 --vec-store ./vec_store --output-dir ./quizzes_refined --seed 0 --refine --refine-rounds 1
```

With `--refine`, each question costs eight model calls instead of two
(four persona reviews, one context retrieval, one synthesis, plus the
two base calls), so keep the count small. Because the seed and week
match step 2, the first six slots are directly comparable draft-vs-final
pairs. The output CSV carries `draft_question`, `draft_answer`, and
`refined` columns, so the refinement's effect is auditable per item.

## 4. Run the evaluation

```bash
python evaluate_quiz.py --quiz-csv quizzes/week_03_quiz.csv --course-config configs/aim2_course.yaml --report-json reports/week_03_report.json
```

```bash
python evaluate_quiz.py --quiz-csv quizzes_refined/week_03_quiz.csv --course-config configs/aim2_course.yaml --report-json reports/week_03_refined_report.json
```

What each section of the report gives you:

| Report section | What to record |
| --- | --- |
| Schema checks | Count of findings (ideally zero) and what kinds. |
| Bloom's distribution | Counts per level and the normalized entropy. |
| Near-duplicates | Number of pairs at the 0.85 threshold. |
| Coverage | Concepts/readings from the week with zero questions. |
| Draft vs final | (Refined run only) similarity and length deltas. |

Also emit the instrument for the human study while you are here:

```bash
python evaluate_quiz.py --quiz-csv quizzes/week_03_quiz.csv --emit-rating-sheet reports/week_03_ratings_blank.csv
```

## 5. Fold results into the README

1. **Example output.** Copy one representative CSV into `examples/`
   (the `.gitignore` already whitelists `examples/*.csv`):

   ```bash
   mkdir -p examples && cp quizzes/week_03_quiz.csv examples/week_03_quiz_example.csv
   ```

   Read it first, end to end. Everything in it becomes public.

2. **Sample questions.** Pick 2-3 Q&A pairs spanning different Bloom's
   levels for the Output schema section -- include at least one you
   consider imperfect, with a sentence on why. A visibly honest example
   is worth more than three cherry-picked ones.

3. **Measured numbers.** Add a short "Measured results" subsection under
   Evaluation and limitations reporting only what step 4 printed: parse
   success rate, Bloom's distribution and entropy, duplicate count,
   coverage gaps, and the draft-vs-final deltas. State the generation
   settings (weeks, counts, seed, model, date) so the numbers are
   reproducible, and keep the existing caveat that these are structural
   checks, not quality measurements.

4. Re-run `python -m pytest -q` (the repo hygiene test also walks new
   files), then commit the README change, the example CSV, and the JSON
   reports together.

## 6. The human evaluation (the result that actually matters)

Everything above is automatable and therefore weak evidence. The
credible result is the blinded expert rating described in the README's
protocol:

1. Generate ~60 questions stratified across weeks and all six levels
   (`--num-questions 60` over 3-4 weeks, or 10-12 per week across five
   weeks).
2. Emit one blinded rating sheet per rater with
   `--emit-rating-sheet`; recruit 2 raters (instructors or teaching
   fellows who know the readings).
3. Raters fill in `rated_bloom_level`, `factually_correct`,
   `answerable_from_reading`, and `would_use` without seeing the
   requested level.
4. Analyze: the 6x6 requested-vs-rated confusion matrix, quadratic
   weighted Cohen's kappa, rater-vs-rater agreement as the ceiling, and
   the proportions for the three yes/no columns. This directly tests the
   README's stated hypothesis that Analyzing/Evaluating requests
   collapse toward Understanding.
5. Report whatever comes out, including a null or negative result --
   for the audience this repository is aimed at, a real negative result
   with a correct analysis is stronger than an unsupported positive
   claim.

Only after step 4 exists as human labels is it worth implementing the
`score_groundedness` / `score_bloom_alignment` stubs, because only then
can an LLM judge be calibrated against something.

## Cost notes

Ballpark call counts for the full sequence above: index build (one
embedding pass over the corpus) + 12 x 2 + 12 x 2 (two baseline weeks)
+ 6 x 8 (refined week) = roughly 100 chat/query calls plus one
embedding pass. At typical gpt-4o pricing this is single-digit dollars;
the 60-question human-study generation adds roughly 120-480 calls
depending on refinement. Check your Azure deployment's pricing before
scaling past that.
