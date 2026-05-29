# Writing Style Reference Guide — Personal Voice Edition

*This document captures the author's personal writing voice. It should be consulted before drafting any document on behalf of this author. The blog posts at armchairquantumphysicist.com are the **primary and most reliable voice reference** — they are self-written and uncontaminated by AI. Technical specification documents (such as hybrid_family_final.md) are partially AI-assisted and should be treated as secondary references for structure conventions only, not for voice. Business communication samples (emails, memos) are the primary reference for that register.*

---

## 1. Who This Person Is as a Writer

The author is a **native Dutch speaker writing in English**, intellectually curious, direct, and opinionated without being aggressive. He builds arguments through questions as much as through statements and is genuinely comfortable leaving questions open at the end of a piece — he does not pretend to resolve what he has not resolved.

**The most important thing to know about his voice:** it thinks out loud. The reader is brought along on the thinking process, not handed a finished conclusion. This is what makes his writing feel alive rather than authoritative.

The three registers, in order of voice authenticity:

| Register | Examples | Trust level for voice |
|---|---|---|
| Scientific blog | armchairquantumphysicist.com | **Primary** — self-written, use as ground truth |
| Business communication | SGi2.0 email, VPC memo | **Primary** for that register |
| Technical specification | hybrid_family_final.md | **Secondary** — partially AI-assisted, use for structure conventions only |

### AI Contamination Warning
The technical specification documents show patterns that are **not** this author's natural voice and should not be reproduced when writing in his voice:
- Overly clean parallel structure in every bullet and table row
- Formal theorem/definition/proof block language
- Phrases like *"it is worth noting that"*, *"the normative definition"*, *"the authoritative definition"*
- Exhaustive enumeration of every edge case in prose
- Zero parenthetical asides — the author's natural writing has many
- No questions anywhere — the author's natural writing asks questions constantly

---

## 2. Voice and Tone

### 2.1 Inclusive "We"
Even when writing alone, the author uses **"we"** as the default subject. This is not the royal we — it is genuinely inclusive, pulling the reader along: *"we allow Alice and Bob to utilize non-local resources"*, *"we have modeled this game in Python"*, *"we see that..."*. Switching to "I" signals a personal opinion or a deliberate shift: *"I wonder whether..."*, *"I think it is wise to consider..."*.

### 2.2 Questioning Tone
The author frequently **thinks out loud through questions**, both rhetorical and genuine. This is one of the most distinctive features of his voice. Questions are not just rhetorical devices — they signal actual uncertainty or an invitation to the reader to think:
- *"Whether these correlations truly indicate non-locality, or whether we can identify alternative mechanisms remains an open debate."*
- *"What story resonates with expecting parents, or new parents, while they stand in front of a shelf?"*
- *"I wonder whether some of the beliefs we have... are real needs, or more internal beliefs."*

### 2.3 Building from Extremes
When introducing a concept or formula, the author grounds it by testing **boundary cases** — 100% and 50%, best case and worst case — before discussing the general case. This is a consistent pedagogical habit:
- *"If he knows the full bitstring, his success rate will be 100%; if he does not know anything about Alice's bits, his success rate will be 50%."*
- *"For a q-value of 1, they behave as regular entangled photons, and for increasing values for q, their behaviour becomes more and more non-local."*

### 2.4 Open Endings
The author does **not close pieces with neat conclusions**. Conclusions acknowledge what has been shown and then explicitly reopen the question or name what remains unresolved:
- *"This conclusion still leaves open the question on non-local correlations... Whether these correlations truly indicate non-locality, or whether we can identify alternative mechanisms remains an open debate."*
- *"Possibly this means that Information Causality is one of the governing principles of physics..."* — note "possibly", not "therefore".

### 2.5 Frank, Direct Opinions in Business Writing
In business contexts the author is **frank and direct without being harsh**. He names problems clearly, questions assumptions others hold as given, and makes concrete proposals. He does not soften criticism with excessive diplomatic framing:
- *"I wonder whether some of the beliefs we have... are real needs, or more internal beliefs."*
- *"By innovating on the nipple we do make our life difficult."*
- *"Somehow in the project we have to overcome this drawback."*

---

## 3. Characteristic Phrases and Constructions

These phrases appear repeatedly across multiple blog posts and should be reused where natural. Do **not** replace them with synonyms for variety. Phrases marked ★ appear in at least two of the three blog posts analysed and are the most reliable fingerprints.

| Phrase / Construction | Where it appears | Notes |
|---|---|---|
| ★ *"This triggers the question..."* | Blog, after introducing a result | One of the most distinctive phrases; signals the pivot from fact to implication: *"This triggers the question of why nature limits non-locality"* |
| ★ *"So, ..."* | Transitions and conclusions | Used multiple times per post as a connector: *"So we see that..."*, *"So, in our thought experiment..."* |
| ★ *"Note that..."* | Soft parenthetical in prose | Draws attention to a non-obvious property without breaking flow: *"Note that the probability only depends on the difference between the angles."* |
| ★ *"Specifically, ..."* | Narrowing scope | Precedes a concrete example or constraint |
| ★ *"...more and more..."* | Gradual intensification | *"more and more non-local"* — prefers this over superlatives |
| ★ *"In this post, we will discuss..."* | Second paragraph of blog posts | Never the opening sentence — always appears after the framing paragraph |
| ★ *"What do we expect to find?"* | Before experimental predictions | Always followed by a numbered list — this pairing is very consistent |
| *"Interestingly enough, ..."* | Blog openings, transitions | Signals a surprising or counterintuitive point |
| *"On the other hand, ..."* | Contrasts | Standard contrast marker |
| *"Of course, ..."* | Concessions | Acknowledges the obvious before moving on |
| *"...etc etc"* | Informal lists | Double "etc" — characteristic, retain as-is |
| *"whether we can conclude..."* | Epistemic caution | Signals genuine uncertainty |
| *"What would go wrong if..."* | Blog, after establishing a limit | Frames the question of why a bound exists: *"What would go wrong if Nature exceeds the Tsirelson bound?"* |
| *"We can also consider..."* | Generalisation transition | Moves from specific to general, or introduces a parameter sweep |
| *"Around the same time..."* | Historical narrative | Time-anchors parallel developments |
| *"In practice..."* / *"in principle..."* | Distinguishing theory from reality | Used as a pair to contrast what is possible in theory vs. experimentally |
| *"I wonder whether..."* | Personal reflection (business) | Marks a shift to genuine questioning |
| *"The proverbial..."* | Archetypes (business) | *"the proverbial Dad sent to Walmart"* |
| *"nr 1"* | Rankings in informal writing | Dutch-style abbreviation, retain in business writing |

---

## 4. Sentence Structure and Length

### 4.1 Typical Sentence Length
Sentences are **medium length** — typically 15–25 words. The author does not write telegraphically, but also avoids the long, clause-heavy constructions common in academic German-tradition writing. Sentences build one idea at a time.

### 4.2 Sentence Patterns
The author favours these constructions:

**If...then structure for intuition building:**
> *"If he knows the full bitstring, his success rate will be 100%; if he does not know anything about Alice's bits, his success rate will be 50%."*

**Semicolon to pair two related statements:**
> *"If *P(β|α)* = 1, we always detect photon two in polarization *β*...; if *P(β|α)* = 0, we never see the second photon..."*

**Adverb-comma-subject opening:**
> *"Specifically, they proposed a probability distribution where..."*
> *"Interestingly enough, the debate... is as lively as ever."*
> *"Surprisingly, Pawlowski bases his arguments on information theory rather than on arguments from physics."*

**"So" as mid-paragraph connector:**
> *"So, Alice receives a bitstring as an assignment, and Bob gets an index."*
> *"So, in our thought experiment, we can consider any object..."*

**"Note that" as soft aside:**
> *"Note that the probability only depends on the difference between the angles."*
> *"Note that individually, these photons still behave as 'normal' photons."*

**Parenthetical "i.e.,":**
> *"a local theory as a theory where the likelihood of events occurring (i.e., the possibility to observe specific measurement results) purely depends on the direct environment."*

**Numbered list for experimental expectations**, always preceded by *"What do we expect to find?"*:
> *"What do we expect to find?*
> *1. If the half-wave plates are aligned, we expect perfect correlation...*
> *2. We expect that the shape of the correlation curve changes...*
> *3. If we change the detection for the first photon..."*

**"Suppose..."** to set up a thought experiment scenario:
> *"Suppose the events occur at a large enough distance, and the time difference between the events is small enough."*

**Historical anchoring:**
> *"Around the same time that Alain Aspect published his results..., Boris Tsirelson published his work..."*
> *"Since 1935, a discussion has been going on in Physics on non-locality..."*

### 4.3 Paragraph Rhythm
Paragraphs are typically 3–5 sentences. In the blog, they are slightly shorter and more conversational. In technical specifications, they are denser. In business writing, they can be as short as 1–2 sentences when making a direct point.

---

## 5. Punctuation Habits

### 5.1 Single Quotes for Technical Terms
When introducing a concept or using a term in a non-standard or contested sense, the author uses **single quotes** — not double quotes, not italics:
- *'local'*, *'non-local'*, *'useless non-locality'*, *'useful non-locality'*, *'superphotons'*, *'factions'*

This is consistent and distinctive. Use single quotes for this purpose.

### 5.2 Parentheses for Asides
The author uses **parentheses frequently** for supplementary information, cross-references, and qualifications. This is a signature of his writing:
- *(i.e., the possibility to observe specific measurement results)*
- *(we discussed this game in an earlier post)*
- *(see the Jupyter Notebook on GitHub for more details)*
- *(if so, what group)*
- *(the consumer)*

### 5.3 Ellipsis for Trailing Thoughts
In informal and business writing, the author uses **"..."** to trail off into an open implication. This is common in Dutch writing and is part of his voice in those contexts:
- *"which can lead to new listings, publicity etc etc..."*
- *"Maybe a bottle in a nice new color with get you more traction than a nipple..."*

Do not remove these in informal contexts. In scientific blog writing they appear less often.

### 5.4 Emoji
Used sparingly in business writing for warmth, never in scientific writing:
- *"I think that if you would put on the box in big letters 'this nipple is guaranteed to work for your baby, if not 10x money back' you would blow the market 😊"*

---

## 6. Non-Native Speaker Patterns

The author is a native Dutch speaker. The following patterns are characteristic of his English and should be **preserved** (they are part of his voice, not mistakes to correct, unless he explicitly asks for proofreading):

| Pattern | Example | Dutch origin |
|---|---|---|
| *"te"* instead of *"the"* | *"Somehow in te project"* | Direct Dutch slip (*"het project"*) |
| *"is send"* instead of *"is sent"* | *"the proverbial Dad who is send to Walmart"* | Dutch passive (*"wordt gestuurd"*) |
| Inverted word order | *"We do focus in our VPC on the end-user"* | Dutch V2 word order |
| *"nr 1"* for *"number 1"* | *"the nr 1 concern"* | Dutch abbreviation style |
| Double "etc" | *"publicity etc etc"* | Dutch habit |
| Slightly formal phrasing | *"the amount of information potentially available to Bob"* | Translated feel |
| *"Pls"* abbreviation | *"Pls let me know"* | Common in Dutch business English |
| Missing article occasionally | *"for guessing a bit, a success rate of 50%"* | Article drop |

When **proofreading** is explicitly requested, correct the grammatical errors (is send → is sent, te → the) but preserve stylistic choices (double etc, single quotes, parenthetical frequency).

---

## 7. Structure and Document Conventions

### 7.1 Scientific Blog Structure
- Named sections with `##` headers
- Numbered references in text as [1], [2] — collected at end
- Figures referenced inline and described in captions in italics
- Code and GitHub linked for reproducibility
- Cross-references to earlier posts: *"(see also our earlier post)"*
- Conclusion section always present, always opens back up rather than closes down

### 7.2 Technical Specification Structure
- Numbered sections with subsections (`## 1.`, `### 1.1`)
- Notation table near the top
- Horizontal rules between major sections
- Verbal description → formal definition → pseudocode → interpretation
- Both A-side and B-side described separately for every protocol
- Parameter boundary tables
- Key results boxed

### 7.3 Business Communication Structure
- Opens with direct statement of situation or problem
- Bullet list of open questions or action items
- Closes with direct statement of next step
- Short, action-oriented paragraphs
- No preamble or diplomatic softening before the main point

---

## 8. What Makes This Voice Recognisable — Summary

If you were to pick this author's writing out of a lineup, these are the signals, in rough order of distinctiveness:

1. **"This triggers the question..."** — the single most characteristic phrase; appears in multiple posts at exactly the same structural moment
2. **Two-move opening** — broad framing first, *"In this post, we will discuss..."* always in the second paragraph, never the first
3. **Single quotes around contested or informal terms** — *'non-local'*, *'superquantum'*, *'spooky action'*, *'useless non-locality'* — never italics, never bold
4. **Numbered list for predictions, always preceded by "What do we expect to find?"** — this exact pairing appears in multiple posts
5. **Historical narrative before technical content** — scientists named, years given, sequence of discovery told before the formalism
6. **Frequent parenthetical asides** — the reader feels accompanied, not lectured
7. **Questions left genuinely open at the end** — *"possibly"*, *"remains an open debate"* — never over-claims
8. **"We" throughout** — inclusive, never the authorial "I" except for personal opinion
9. **"Note that..."** as a soft mid-paragraph signal
10. **Toy model introduced first, then generalised** — always bottom-up, never top-down
11. **"So, ..."** as the most common connector
12. **Boundary cases to build intuition** — 100%/50%, q=1/q=∞, before the general formula
13. **Dry parenthetical humour** — *(See [6] for the banana reference)*
14. **Philosophical but modest conclusions** — *"Possibly this means..."*, not *"We have proven..."*
15. **Python and GitHub always linked** — open science is a consistent value, not an afterthought

---

## 9. Real Voice vs. AI Contamination — How to Tell the Difference

This section exists because the author's technical documents are partially AI-assisted. The contrast between his real voice (blog) and the AI-contaminated output (specification) is instructive.

| Feature | Real voice (blog) | AI contamination (spec) |
|---|---|---|
| Parenthetical asides | Frequent: *(i.e., ...)*, *(see earlier post)* | Rare or absent |
| Questions | Multiple per section, genuinely open | Absent |
| Conclusions | *"Possibly..."*, *"remains an open debate"* | *"This completes the proof"*, definitive |
| Term introduction | Single quotes: *'non-local'* | Bold or italics: **non-local** |
| Sentence variety | Mix of short and medium, some semicolons | Consistently medium, very parallel |
| Connectors | *"So, ..."*, *"Of course, ..."*, *"On the other hand"* | *"Note that"*, *"It follows that"*, *"Moreover"* |
| Intuition building | Boundary cases, concrete numbers, extremes | Abstract general case stated first |
| Hedging | *"possibly"*, *"it appears that"* — genuine | Either absent or formulaic |
| Cross-references | *"we discussed this in an earlier post"* — conversational | *"see Section 4.2"* — mechanical |
| Ending | Opens a new question | Summarises what was shown |

When writing on behalf of this author, default to the left column. The right column is what to avoid.

---

## 10. Blog Post Conventions (Primary Reference)

Since the blog is the ground-truth voice, these conventions are canonical. All three posts confirm the same structure.

### 10.1 Opening Pattern (confirmed across all three posts)
The opening is **never** "In this post, we will..." — that line always appears in the **second paragraph**, after a framing opening. The first paragraph establishes the big-picture context or question. This two-move opening is consistent:

- *Move 1 (paragraph 1):* A broad, surprising, or historically grounded framing statement. May be a question. Does not mention the post's specific content yet.
- *Move 2 (paragraph 2):* *"In this post, we will discuss..."* — names the specific work or topic being covered.

**Example from information causality post:**
> Para 1: *"Can we explain fundamental physics if we use 'information' as the basic concept instead of the usual concepts like matter and energy? Some physicists argue that this is indeed the case for quantum mechanics."*
> Para 2: *"Since 1935, a discussion has been going on... In this post, we will discuss the work of Marcin Pawlowski..."*

### 10.2 Section Headers
Named with `###`, descriptively, never numbered within a post. Headers are short — typically one or two words naming the concept: *"CHSH"*, *"Superquantum"*, *"The guessing game"*, *"Entangled photons"*, *"Superentangled photons"*. No "Introduction" or "Conclusion" as explicit headers — the conclusion is just the final section written as a paragraph.

### 10.3 Historical Narrative
New concepts are almost always grounded in their **historical development** before the technical content appears. The author names scientists, dates, and the sequence of discoveries. This is not background padding — it is part of how he makes ideas feel earned rather than dropped in.

### 10.4 Toy Model First, Then Generalise
A simplified model is always introduced before the general case: *"The toy model used when studying non-locality consists of two photons entangled in their polarization state."* Generalisation follows: *"We can also consider a gradual transition... We introduce a 'superquantumness' parameter q..."*

### 10.5 Numbered Lists for Predictions Only
Numbered lists appear in one specific context: **before running an experiment or simulation**, preceded by *"What do we expect to find?"*. They are not used for general enumeration — that stays in prose. This pairing (question + numbered list) is one of the most reliable structural fingerprints.

### 10.6 Code and Open Science
Python code and GitHub are always mentioned and linked. This is not incidental — it reflects a genuine commitment to reproducibility. The FockStateCircuit package appears by name. Links go to both the GitHub repo and GitHub Pages.

### 10.7 Figures
Figures are embedded with captions in italics. The caption describes what is shown and sometimes interprets it briefly. Figures are always referred to in the prose before they appear: *"In Figure 3, we plot..."*, *"This picture summarises the setup."*

### 10.8 Numbered References
References appear as [1], [2] etc. inline in text, collected in a numbered list at the end. Never author-year format in blog posts.

### 10.9 Conclusion Pattern
The conclusion section is written as a paragraph (no header needed, or a simple "Conclusion" header). It states what was shown, then **pivots to what remains open**. It never over-claims. The word *"possibly"* or *"it appears"* typically signals the conclusion's main claim:
> *"Possibly this means that Information Causality is one of the governing principles of physics..."*
> *"Whether these correlations truly indicate non-locality, or whether we can identify alternative mechanisms remains an open debate."*

### 10.10 Humour
Humour appears occasionally, always via parenthetical: *(See [6] for the banana reference)*. It is dry, light, and never forced.

---

## 11. Quick Checklist Before Submitting a Draft

**Voice:**
- [ ] Single quotes (not italics or double quotes) around contested or newly introduced terms
- [ ] Parenthetical asides present where supplementary context is natural
- [ ] "We" as default subject, "I" only for personal opinion or reflection
- [ ] At least one genuine question somewhere in the piece
- [ ] Boundary cases used when introducing formulas or concepts
- [ ] Conclusion leaves something open — does not over-claim
- [ ] "So, ..." used as a connector at least once in longer pieces
- [ ] "Possibly" or "it appears" rather than "therefore" or "we have shown" at conclusions

**AI contamination check — none of these should appear:**
- [ ] No *"it is worth noting that"*
- [ ] No *"it follows that"* or *"moreover"* as connectors (use "so" or "and")
- [ ] No *"the normative definition"* or *"the authoritative"* anything
- [ ] No conclusions that summarise and close — always reopen
- [ ] No perfectly parallel bullet lists for conceptual content
- [ ] No absence of parenthetical asides across a full section

**Conventions:**
- [ ] Numbered references [1], [2] in text, collected at end (blog/scientific)
- [ ] Non-native patterns preserved unless proofreading explicitly requested
- [ ] Cross-references written conversationally, not mechanically
