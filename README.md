# NFIP Claims: Data Integrity and ML Readiness

Take-home exercise.
Deadline: Friday 9 October 2026, 10:00 AM ET.

This exercise covers the full path from a raw data file to a working model: clean
the records, build a simple model, and serve it behind a small portal. The steps
connect, so choices you make while cleaning affect what you can model later.
Treat it as one connected task, not separate pieces.

This is deliberately more than you may finish. Do the core tasks first, then take
the stretch items if you have time. A few things done well beats everything done
quickly. We care more about your judgement, and how clearly you explain it, than
about finishing every item.

## Data

`claims.csv` has about two thousand US flood insurance claims from the National
Flood Insurance Program (NFIP). The records come from more than one source system
and have not been cleaned. Field names follow the OpenFEMA FIMA NFIP Redacted
Claims schema (search that name for field definitions).

## Core tasks

### 1. Clean the data

Go through the dataset and find values that look wrong, inconsistent, or
impossible. Decide what to do about each, and apply the fix. We want to see the
problems you found and your reasoning, not just a quietly cleaned file.

Write each issue up in a few lines. Use this format:

    Problem:  short description of what is wrong with a field.
    Evidence: one or two example claim_refs, and roughly how many rows.
    Fix:      what you did, in one line (or why you chose to leave it).

### 2. Frame and build a model

Pick one predictive task this data can support. For example: predict the building
claim payment, or flag claims that look anomalous. Prepare the data, train a
simple baseline, and report the result in plain numbers.

State it plainly. For example:

    Task:   predict building claim payment.
    Split:  how you split train vs test, and why.
    Result: test MAE about $X across Y claims.
    Caveat: what you are unsure about, or what could break this.

A clean, honest baseline is worth more than a complex model you cannot explain.

### 3. Build a running portal

Put the above behind a small web page with a backend. A user gives it an input
and gets your output back. Running on your own machine is fine.

For example, one interaction:

    In:  a single claim (pasted fields, or an uploaded row).
    Out: the cleaned version of that claim,
         the model's prediction (for example, predicted building payment = $X),
         and any data-quality flags you raised on that row.

Keep it plain. Function matters more than how it looks.

Throughout, commit as you go with clear messages, and open a pull request.

## Stretch (nice to have, if you have time)

- Deploy the portal to a public URL (Vercel, Render, Railway, a container host,
  or similar) that we can open and use.
- Add basic safeguards: check the input is valid, and limit how often the portal
  can be called (rate limiting).
- Add a system diagram showing the parts and how data moves between them.

## Tech choices and scaling (explain in your writeup)

Use any language, framework, and database you want. Tell us what you picked and
why, including where you store the claims and the model's inputs and outputs.

Then explain, in words, how your design holds up under load. You do not need to
run a load test; we want your reasoning. Cover:

- How the portal serves many requests at the same time.
- What changes as usage grows from 1 to 2, 5, 10, and 100 users at once.
- Which step is the bottleneck (reading data, or running the model), and why.
- What you cache, where, and when the cache is cleared.
- What happens when a free-tier host runs low on compute: does it slow down,
  queue, reject extra requests, or crash, and what would you want it to do?
- Where specific data structures or algorithms would speed things up: name the
  part, the technique, and the gain. For example, an index or hash map for
  lookups instead of scanning, a heap for a top-N query, batching requests to the
  model, or vectorised operations instead of row-by-row loops.

For example, the level we are looking for:

    "One API process, one model worker. Reads are cheap; the model runs one
     request at a time, so it is the bottleneck. At about ten users I cache
     repeat inputs (hash of the input to result) and queue the rest so they wait
     instead of timing out. On the free tier I would reply 'busy, try again'
     past N users rather than crash."

## Deliverables

- A running portal: a public URL if you deployed it, or steps to run it locally.
- A repository or pull request link with a readable commit history.
- A short report or a few slides covering: the issues you found and fixed, your
  model and its numbers, your tech choices, and your scaling and optimisation
  answers.
- Steps for us to reproduce your run.
- Your coding agent session, if you used one (see "Using AI tools").
- If you did any stretch items: the deployed URL, the diagram, and a note on your
  safeguards.

## What "done" looks like

Before you send it, check you have:

- [ ] a list of data issues, each with evidence and a fix
- [ ] a model with a stated task, split, and honest numbers
- [ ] a portal that takes an input and returns an output
- [ ] a short report or slides, including your scaling and optimisation answers
- [ ] steps to run it
- [ ] your tool disclosure and coding agent session, if you used one

## How we assess

We look at how you reason about messy data, whether your train/test split is
sound, your git and pull request hygiene, how clearly you explain your choices
and limits, and your scaling and optimisation reasoning. We also check the portal
runs and returns sensible output. Read the data closely: some records have
problems that are not obvious on a first read.

## Using AI tools

You may use AI tools. If you use a coding agent (Claude, ChatGPT, GLM, DeepSeek,
Gemini, or similar), through a chat or CLI interface or inside any IDE, attach
your coding agent session for this task, and say which tools you used and how.
Background on attaching a session:
https://www.reddit.com/r/ycombinator/comments/1qtgw10/coding_agent_session_thats_such_great_question/
