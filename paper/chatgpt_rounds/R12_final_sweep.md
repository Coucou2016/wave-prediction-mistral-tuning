# Round 12 — Final Sweep: Engineering Trace Removal and Submission Readiness

**Role:** Final copy editor. Perform a submission-ready sweep.

**URLs**
- Manuscript (raw): https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md
- Prior rounds: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/chatgpt_rounds

## Instruction

Read the final manuscript and check for:

### 1. Engineering Trace Removal
- No local file paths (e.g., `d:\Projects\...`, `C:\Users\...`)
- No script filenames (e.g., `curve_jsonl_export.py`, `train_lora.py`)
- No conda/pip environment names
- No Cursor/ChatGPT/IDE process references
- No "ADOPT"/"REJECT" metadata
- No internal version numbers or round numbers
- No "TODO" or "FIXME" markers

### 2. Manuscript Metadata
- Target journal line: should read cleanly as a submission note, not a process note
- One-sentence argument: should be the paper's thesis, not a development note
- Title and alternate title: should be publication-ready
- Author list: placeholder or ready?

### 3. Final Readability
- Read the entire manuscript aloud (mentally). Flag any sentence that sounds awkward.
- Check for any remaining typos or formatting errors
- Verify all section numbers are sequential
- Verify all cross-references are correct

### 4. Submission Checklist
- Abstract within word limit (typically 150-250 words for OE)
- Keywords present (6-8)
- Highlights present (4, ≤85 chars each)
- References complete
- Data Availability statement present
- Figure captions present
- No missing sections

### Desired Output Format
```
CRITICAL (must fix before submission):
- ...

MAJOR (should fix):
- ...

MINOR (nice to have):
- ...

ADOPT:
- ...

REJECT:
- ...

FINAL VERDICT: [READY / NOT READY] for submission
```