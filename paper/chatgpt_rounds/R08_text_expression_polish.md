# Round 8 — Text Expression: Academic English, Nature-Leaning Style, Redundancy Reduction

**Role:** Academic language editor. Perform a section-by-section English polish of the manuscript.

**URLs**
- Manuscript (raw): https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md
- Prior rounds: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/chatgpt_rounds

## Instruction

Read the full manuscript. For EACH section, identify and fix the following issues:

### 1. Academic English Norms
- Awkward or non-idiomatic phrasing
- Tense inconsistencies (past vs present for results vs methods)
- Article usage (a/an/the)
- Preposition errors
- Subject-verb agreement

### 2. Nature-Leaning Style (even though targeting Ocean Engineering)
- Prefer declarative, concise sentences
- Avoid "Interestingly," "Notably," "It is worth noting that"
- Use hedging appropriately: "may," "suggests," "appears to" rather than weak "could perhaps"
- Avoid "very," "extremely," "highly" unless essential
- Prefer active voice where natural

### 3. Sentence Variety
- Flag any sequence of 3+ sentences with the same opening structure
- Flag overly long sentences (>40 words) that could be split
- Check for monotonous "The X is/was Y" patterns

### 4. Redundancy
- Flag repeated phrases across sections (especially Intro/Discussion overlap)
- Flag redundant qualifiers ("generally," "typically," "in general")
- Flag sentences that restate the obvious

### 5. Section-Specific Issues
- **Abstract:** Check word count; ensure every sentence earns its place
- **Introduction:** Check funnel structure (broad → gap → this paper)
- **Related Work:** Check for fair representation of prior work
- **Methods:** Check for precision and clarity
- **Results:** Check for neutral reporting (no interpretation in Results)
- **Discussion:** Check that it interprets rather than restates
- **Limitations:** Check that each limitation is honest and specific
- **Conclusions:** Check against Abstract for consistency

### For each section, provide:
1. The polished version of problematic sentences
2. A brief explanation of the change
3. Severity: CRITICAL / MAJOR / MINOR

### Desired Output Format
```
## Abstract
- [MAJOR] Original: "..."
  Fixed: "..."
  Reason: ...

## Section 1: Introduction
- [MINOR] Original: "..."
  Fixed: "..."
  Reason: ...

(continue for all sections)

## ADOPT (changes to make):
- ...

## REJECT (suggestions to skip):
- ...
```