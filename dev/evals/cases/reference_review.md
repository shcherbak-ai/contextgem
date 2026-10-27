# Review of the Reference Answers

The reference answers of `complex_nda` and `complex_dpa` were reviewed on 2026-10-11 (the sections up to "Limits"). `complex_mcu` was written with its reference answers on 2026-10-15, and `complex_stability` on 2026-10-20, and each was reviewed in the same way (their own sections, at the end).

The reference answers of the two contract cases (`complex_nda.gold.json`, `complex_dpa.gold.json`) were produced by an AI agent. This is the record of their review, which was also done by AI agents: no person has verified the answers. The maintainer delegated the review.

## Method

1. **Blind annotation.** For each case, a separate agent got only the text with numbered paragraphs and the definitions of the aspects and concepts. It saw neither the reference answers nor any model result. It listed the paragraphs of each aspect and derived the answer of each concept.
2. **Own derivation.** The reviewing agent read both contracts in full and derived every typed concept answer from the text before comparing it with the reference.
3. **Comparison of three sources**: the reference, the blind annotation, and the five decision models of the baseline of 2026-10-11 (`baseline_2026-10-11`; how many of them give a paragraph a probability of 0.5 or more for an aspect).
4. **Ruling.** Every paragraph on which the reference and the blind annotation differ, and every paragraph on which four or five decision models disagree with both, was ruled on against the text.
5. **Second part**, after the baseline of 2026-10-11 was recorded again with runs that keep what they extract: the evidence of each concept, the justifications and the string answers were reviewed, and what the hosted LLMs extract beyond the reference (seven runs) was ruled on in the same way.

The standard for an aspect is the one the models are given: a paragraph belongs to an aspect if it clearly relates to the aspect, based on its name and description; the context of the surrounding paragraphs counts; tangential mentions are left out.

## Concepts

All three sources agree on every typed concept answer: 18 in the NDA case and 22 in the DPA case (dates, numbers, Booleans, labels and the objects of the JSON object concepts). The string concepts agree in substance; their wording was not compared.

Two readings that the blind annotation flagged as arguable, and how the reference stands on them:

- **Qualifying Fees of LC-27.** B5.1 excludes service credits and also says that a paid credit reduces the fees for its service period. The reference deducts the credit (EUR 1,140,000, a security cap of EUR 2,280,000), which uses both sentences. Without the deduction the cap would be EUR 2,400,000; the total payable is the same either way.
- **Permitted uses of the held ORD records on 1 September 2027.** Both holds have ended and the records are in the deletion window that follows the release. The reference gives "legal preservation" as the only use, as B3.3 keeps the hold and that window as the sole retention authority.

### Evidence, justifications and string answers

- **Evidence** (the paragraphs that the items of a concept reference): compared for each of the 34 concepts that ask for references with the paragraphs that the blind annotation relied on and with those that the hosted LLMs referenced in seven runs. One list was incomplete and is corrected: "Authorised data locations on 15 May 2027" (DPA) had the change records and the rules of Amendment No. 2, but not the paragraphs that establish the authorisations of four of its five countries. Added: [52] 11.3 (storage in Norway and Estonia), [82] and [84] (the Annex 3 approvals in Estonia and Lithuania) and [102] A4.2 (the approval in France). The other lists stand: each paragraph in them is used for its answer, and where the other sources add a paragraph, it supports the answer without being needed for it.
- **Justifications**: each was read against the text and the derivation of its answer; none is wrong. The harness does not score them.
- **String answers**: the five reference values are correct. The harness counts a string answer only if it equals the reference value or an accepted one, apart from the other concepts.
- The 11 concepts of the DPA document as a whole ask for no references, so their evidence is empty by design.

## Aspects

The reference and the blind annotation agree on the substance of every aspect. Their differences are of three kinds.

### Added to the reference answers

| Case | Aspect | Paragraph | Reason |
|------|--------|-----------|--------|
| DPA | Amendment effectiveness and precedence | [126] B3.2 | Its last sentence is an express conflict rule: Section B3 of Amendment No. 2 controls over the retention schedules of Annex 2 of the DPA. The aspect asks for the provisions on how such conflicts are resolved. |
| DPA | Liability LC-27 | [58] 13.2 | The aspect includes "earlier cap terms and carve-outs". 13.2 is the original carve-out (liability to Data Subjects is not limited), next to the original cap of 13.1, which the reference already had. |

Nothing was removed from the reference answers, and the NDA file is unchanged.

### Not scored: section headings

The blind annotation included the heading of each section whose clauses it included; the reference included no heading; the decision models include headings often. A heading has no rule or fact of its own, so extracting it with its section, or leaving it out, is not an error. The heading of a section that has a paragraph of the reference answers of an aspect is therefore not scored for that aspect. Headings of other sections are scored as before.

This matters for the decision models: of the paragraphs that they extract and the reference does not have, between a quarter and two thirds were section headings, depending on the model and the case (e.g. 41 of 71 for the model in Microsoft Foundry on the DPA case, at a threshold of 0.5).

### Not scored: borderline paragraphs

Paragraphs on which the careful readers disagree, and where the text supports both readings. They are listed with the reasons in `<case>.neutral.json`:

- **NDA**: [34] 6.4 for information protection; [41] 8.1 for recipient permissions; [4] 1.2 and [5] 1.3 for exit obligations.
- **DPA**: [103] A4.3 for amendment effectiveness; [91] A1.1, [118] B1.1 and [119] B1.2 for breach notification, whose description asks for the clauses and their amendments, not for effective dates and clocks; [92] A1.2 and [93] A1.3, the general continuity and precedence clauses of Amendment No. 1, for the four case aspects; [55] 12.2 for retention; [119] B1.2 for supplier approvals.

### Not scored: linked context

The hosted LLMs, which see the whole document, often add paragraphs that an aspect depends on without being a part of it: the clause with a defined term or a date that a rule of the aspect uses, or a clause elsewhere that refers to the aspect. Such a paragraph gives the fuller picture, which helps when the concepts of the aspect are extracted from its paragraphs; leaving it out keeps to the aspect. Neither is an error, so these paragraphs are not scored for the aspect. They are listed as `linked` in `<case>.neutral.json`, each with the dependency in the text:

- **NDA**: [4] 1.2 (Effective Date) and [5] 1.3 (disclosure window) for information protection; [4] 1.2 (Permitted Purpose) for recipient permissions.
- **DPA**: [60] 14.1, [61] 14.2, [67] the duration line of Annex 1 and [76] the retention of access logs for TR-27; [61] 14.2 for LC-27; [50] 11.1 for supplier approvals.

A candidate was a paragraph that at least four of seven runs of the hosted LLMs, or four of the five decision models, extract beyond the reference; it is listed only if the text shows the dependency. Paragraphs that merely share a word with the aspect stay scored (see below).

### Confirmed as they are

- **Left out, although four or five decision models include them**: paragraphs of another subject that share a word with the aspect. NDA: [36] 7.1, [65] C1.3, [67] D1.1 and [71] D1.5 for information protection, and [67] D1.1 for Q-27. DPA: [137] R1.2 for breach notification (a record of one incident, not a clause); [148] R3.5 for IR-27 (it names IR-27 only as the purpose of an emergency permission).
- **Kept, although at most one decision model includes them**: paragraphs that only relate to the aspect in the context of the document, which a decision model does not see. NDA: [27] 5.1 for exit obligations. DPA: [138] R1.3 for IR-27; [91] A1.1 and [119] B1.2 for TR-27; [32] 6.4, the three entries of Annex 3 ([82] to [84]), [103] A4.3 and [118] B1.1 for supplier approvals.

## Effect on the Scores

The baseline of 2026-10-11 was recorded again after the first part of the review, as the earlier runs had kept their scores but not what they extracted. The runs now keep it, and were scored again from it after the second part (`rescore`). Aspect F1 before the review and after it (NDA / DPA, mean of the runs of a configuration):

| Model | Before | After |
|-------|--------|-------|
| `openai/gpt-6-luna` | 0.956 / 0.899 | 0.986 / 0.968 |
| `openai/gpt-6-luna`, with concurrency | 0.922 / 0.845 | 0.956 / 0.960 |
| `openai/gpt-5.6-luna` | 0.954 / 0.820 | 1.000 / 0.968 |
| `openai/gpt-6.1-sol` | 0.963 / 0.875 | 1.000 / 0.941 |
| `ollama_chat/mistral-small:24b` | 0.778 / 0.566 | 0.771 / 0.638 |
| `openai/gpt-6-luna` as a decision model | 0.827 / 0.696 | 0.866 / 0.832 |
| `typesafe/jev-latest` | 0.786 / 0.634 | 0.824 / 0.791 |
| `azure_ai/ms-decision-model` | 0.693 / 0.610 | 0.781 / 0.771 |
| `strands_decider/kev-4b` | 0.701 / 0.587 | 0.768 / 0.718 |
| `strands_decider/nimble` | 0.615 / 0.542 | 0.692 / 0.697 |

The runs of the LLMs before and after are different runs, so their difference also has the variation between runs in it. The decision models give nearly the same answers in each run: their difference is the review. The concept scores did not change.

What is left between the hosted LLMs and the reference is small. In the NDA they miss nothing and add a paragraph in 5 places over seven runs. In the DPA the one systematic difference is the conflict rule of B3.2, which this review added and six of seven runs leave out.

## Limits

- The review is the work of AI agents. It makes the reference more consistent; it does not make it verified.
- With the reviewed reference, the hosted LLMs are at or near the top of the scale on both cases, for aspects and for concepts. The cases can show that a change makes extraction worse, and how models and settings differ in cost; they have little room left to show that a change makes a capable model better.
- Whether a paragraph "clearly relates" to an aspect has judgement in it. The borderline list is where that judgement was closest.
- A decision model judges a paragraph on its own, while the reference lists what belongs in the context of the document. Part of the difference between the two is this, not an error of either.

## complex_mcu (2026-10-15)

### How the case was made

An AI agent wrote the document and its reference answers together, from one set of facts. The document is a fictional documentation package of a microcontroller: an extract of the datasheet, the errata sheet, a product change notification (PCN) with an amendment, an extract of an application note, and the design review and purchasing records of a customer for two boards, DR-114 and DR-115. A script computed each derived answer from the facts and checks it: the junction temperatures (108.2 °C and 100.4 °C), the clock frequencies, the wait states, the ADC sampling rate, the calendar dates and the ISO week of a date code. No answer is close to a rounding boundary.

The answers depend on the following, which is what the case tests:

- **Later documents override earlier values.** For example, DC-2 corrects a wait-state limit, and the PCN changes a BOR threshold and a retention time for revision C. The errata sheet of September 2026 overrides the PCN of June 2026 on ES-08 and ES-09.
- **The silicon revision is decoded from the marking**: revision C is marked Y.
- **Errata apply only under their conditions**, which the board records meet or not.
- **The application note is informative.** Its thermal shortcut, an example at another frequency, and a laboratory INL would give wrong answers.
- **The records have distractors**: a prototype lot of another revision, a measured temperature, a revised forecast.

### Method

The method of the two contract cases, with two separate agents, each given only the text with numbered paragraphs and the definitions of the aspects and concepts:

1. **Blind derivation of every concept answer**, with its evidence and calculation, and a report of any ambiguity or error in the document.
2. **Blind annotation of the paragraphs of every aspect**, with the paragraphs it was unsure about.

Each difference was ruled on against the text.

### Concepts

The blind derivation reaches the same value on all 26 typed concepts. The one string concept matches in substance, and its wording was added to the accepted values.

The ambiguities it records are the traps of the case, each with the reason why the reference value holds. For example, ES-03 would apply to the prototype lot of revision B, not to the production boards. One more reading: ES-02 applies to DR-115 because its junction temperature can fall below 0 °C at power-up at an ambient of −25 °C, although under full load at that ambient it stays at about +5 °C.

Evidence, i.e. the paragraphs that the items of a concept reference:

- **Added** from the blind derivation:
  - 9.1, the conditions of Table 18, for both junction temperatures;
  - 1.5, the decoding of the marking, for the maximum SYSCLK;
  - II.2, why revision C is not marked C, for the marking code;
  - the older statements of class 2 for ES-08 and of ES-09 as fixed (row 2 of Table E3 and III.3 of the PCN), as superseded values are part of the evidence where an answer depends on the change.
- **Not added:**
  - the order of precedence, which the blind derivation cites for most answers: each later document states its change itself;
  - the column headers of tables;
  - table rows that an answer does not need.

### Errors in the document, found by the blind derivation and corrected

- VDDA had a minimum of 1.8 V, below which the 1.8 V ±3 % rail of DR-114 falls. The minimum is now 1.71 V, also in the INL and DNL conditions of Table 17.
- The application note reported a test at 3.0 Msps with a VDDA of 1.8 V, which the fADC limit of 36 MHz does not allow. It is now 2.57 Msps.
- The ADC was "up to 3.6 Msps", which the formula of 7.2 does not give. It is now 3.57 Msps, also in the ENOB condition.
- TIM1 had "a break input" in 1.2 and two in 13.1. Both now say there is more than one.
- DC-4 gave the current of VOH without its sign.

No answer changed. The derivation also noted two points that stay as they are: the PCN was never amended on ES-08 and ES-09, which the order of precedence resolves, and grade A devices have no maximum IDD or endurance specified above a Tj of 125 °C, which no answer needs.

### Aspects

The blind annotation has every paragraph of the reference answers except two groups:

- **The fREF rows that DR-115 does not use** (PLLxRGE = 00, 01 and 11). Only the row of the range it configures is needed, so these rows are now borderline.
- **Two flash paragraphs:** the row of the errata history that records the adding of DC-2, and the wait-state example of the application note. These are now borderline.

What it adds beyond the reference answers was ruled on as follows:

- **Paragraphs on the borderline or linked lists**, or that it was unsure about, are listed as borderline. These are:
  - the order of precedence (in five aspects);
  - the scope of the errata sheet (II.1);
  - the descriptions of errata that are not present in the revision of a board;
  - the informative paragraphs of the application note;
  - the fADC limits (7.1) for the INL of DR-114.
- **Four paragraphs that I had listed as borderline beforehand are scored**, as it left them out without doubt: row A of Table E1 and II.2 for the DR-114 clock configuration, 3.4 for the DR-114 thermal assessment, and 15.2 for the flash specifications.
- **The descriptions of the four aspects of the boards and of the end-of-life aspect were reworded** after the annotation, as "What is needed to determine …:", followed by the kinds of rules and records. The annotation had taken "All clock, PLL, … rules" in the DR-114 clock configuration literally, with whole tables, while it applied "needed to determine" in the other aspects. The lock-time row of Table 10, which it added on that reading, is not needed and stays scored.
- **Section headings, table captions and column headers are not scored**:
  - a heading or a caption is not scored where its section has a paragraph of the aspect;
  - a column header is listed as linked to the aspect whose rows it heads.

### Second part, after the baseline of 2026-10-15

The method of the second part of the review of the contract cases. A candidate is a paragraph that at least 4 of the 7 runs of the hosted LLMs in the default mode (`openai/gpt-6-luna` 3, `openai/gpt-5.6-luna` 3, `openai/gpt-6.1-sol` 1) extract beyond the reference answers, or leave out of them. A candidate is listed as linked if the text shows the dependency, and as borderline if careful readers disagree on it:

- **Device limitations of revision C, linked:**
  - the rows of Table E2 of the limitations that revision C does not have (ES-01, ES-02, ES-03, ES-06 and ES-07; 5 or 6 of 7 runs), whose column of revision C says so;
  - the descriptions of ES-02, ES-03, ES-06 and ES-07 (6 of 7 runs), which end with the revision that fixed them;
  - the row of revision C in Table E1 (4 of 7 runs), which identifies it.

  As in the contract cases, these give the fuller picture: the list of the limitations of revision C is read against them.
- **DR-114 thermal assessment:**
  - 5.3, which defines the output frequency of a PLL, is now linked rather than in the reference answers. All 7 runs left it out; the calculation depends on it only through the clock configuration of the boards. It is also out of the evidence of the two concepts of the aspect, which can only reference what the aspect has.
  - The rows of Tables 18, 19 and 20 for another frequency, package or ADC clock (4 or 5 of 7 runs) are borderline: the tables are read as a whole.
- **DR-114 errata and firmware, borderline:** the other rows of Table 17 (6 or 7 of 7 runs), as the aspect names "the ADC sampling and accuracy rules", and row 2 of Table E3 (4 of 7 runs).
- **Flash memory specifications, borderline:** 6.2, the sector size and the programming, and row 5 of the revision history of the datasheet. The blind annotation included them, 6 of 7 runs left them out, and the description lists latency, endurance and retention only.
- **From the decision models:** as on 2026-10-11, the paragraphs that the decision models extract beyond the reference answers are candidates too: here those that all 3 hosted decision models extract, 7 in all. Two are borderline for the DR-114 thermal assessment: the absolute maximum Tj of Table 2, a limit that HDR-7 does not use and that a reader checks to rule it out, and DC-3, which corrects 8.2 for a QFN-48 with an unsoldered pad. The other five only share a word with their aspect and stay scored (the caption of Table 2, 3.3 for the thermal aspect, DR-115.4 for the flash aspect, DR-114.8 for the DR-114 clock configuration, and IV.3 for DR-115). This part was added after the first pass, which had taken the candidates from the LLMs only and so made unscored only what LLMs add.
- **Kept as they are:**
  - the row of revision C in Table E1 for the DR-114 clock configuration, and 1.3 for DR-115: 4 of 7 runs left each out, and the answers need them;
  - row 3 of Table E3 for DR-115 (4 of 7 runs), which has no bearing on revision B.

The scores of `baseline_2026-10-15` were computed again after this (`rescore`): the aspect F1 of the hosted LLMs on the case went from 0.85-0.94 to 0.90-0.96, and that of the decision models from 0.55-0.61 to 0.57-0.65. No concept score changed.

### Limits

- The case and its reference answers are the work of AI agents. The blind review makes them consistent; it does not make them verified.
- The aspects of this case have more neutral paragraphs than those of the contract cases: from 7 to 32 per aspect, 155 in all, against 55 in `complex_dpa`. Of these, 61 are headings, captions and column headers of the many tables, 69 are borderline and 25 are other linked context. The aspect scores are computed over fewer paragraphs.
- The evidence of many concepts lists every input of a calculation and the superseded values that an answer depends on, while the LLMs reference the governing paragraphs: the reference F1 of the hosted LLMs is 0.64 to 0.77 on this case, against 0.83 to 0.94 on the contracts. Read it to compare models and changes, not as a measure of their references.

## complex_dpa: the wording of two justifications (2026-10-15)

When the judge was added, two reference justifications of `complex_dpa` said that the 24-hour notification limit is "effective on 1 November", for a limit that A1.1 puts in force from 1 October 2026. The judge then found the justifications of the LLMs that say "from 1 October" wrong, in 4 runs. Both are now worded as in force from 1 October, and so on 1 November ("Breach notification deadline in force on 1 November 2026" and "Breach notification obligation summary"). No answer changed, and the scores do not depend on the justifications.

## complex_stability (2026-10-20)

### How the case was made

An AI agent wrote the document and its reference answers together, from one set of facts, to measure what the structure of a document gives the models (see "Extraction from Docling Documents" in `dev/evals/AGENTS.md`). The document is a fictional stability study report of a tablet: three batches in three container closure systems, a release and a shelf-life specification, and a table of results for each batch at each storage condition.

The answers depend on the following, which is what the case tests:

- **Tables that look alike.** The release and the shelf-life specification list the same tests with other limits. The tables of results have the same tests, and the long-term and accelerated ones the same number of columns, with other time points: a row tells its batch, its condition and its time points only through the caption and the header of its table.
- **The concepts of an aspect are extracted from the paragraphs of the aspect only**, where a row can come without the caption and the header of its table: e.g. the 6-month total impurities of V20-2604 at the accelerated condition, among the rows of its three tables.
- **The narrative contradicts the tables**, which the document says record the measured results: it misses two of the five results outside the shelf-life specification (the 3-month water content of V20-2604, and the 6-month impurity D of V20-2607 at the accelerated condition), dates the significant change of V20-2604 at 6 months instead of 3, says that V20-2607 shows no significant change, and proposes 24 months for the Alu/Alu blister, which the rules of section 6 do not allow (12 months, by 6.3).

The results outside the specification, the significant changes and the shelf lives that the rules allow were recomputed by script from the tables.

### Method

As for `complex_mcu`, one separate agent got only the text with numbered paragraphs and the definitions of the aspects and concepts, derived every concept answer with its evidence, listed the paragraphs of every aspect (apart from the headings, the header rows of tables and the paragraphs it was unsure about), and reported every error and ambiguity that it found. Each difference was ruled on against the text.

### Concepts

The blind derivation reaches the same value on all 18 concepts, the two string concepts in substance. It found every planted contradiction, and says for each concept of an aspect that the answer is in the paragraphs of the aspect.

Two definitions it found ambiguous stay as they are, as their descriptions decide them: the shelf life of the Alu/Alu blister is the one that the rules of section 6 allow "applied to the results in the tables, whatever the report proposes" (12 months, not the proposed 24), and the date on which V20-2604 was placed at the intermediate condition is that of Table 1 (3.1), not the start of its testing there. A third was made precise: whether every result outside the specification is investigated now reads "each result in the tables", as the narrative alone would give True.

### Errors in the document, found by the blind derivation and corrected

Three errors that were not planted, corrected in the text; no answer changed:

- 6.5 said that no batch is expected to fall outside the shelf-life specification within 24 months, which the water content of V20-2604 at the long-term condition contradicts (4.0 % at about 18 months on a straight line). It now says it of the assay only, which its decrease of no more than 0.15 % per month supports.
- 3.2 tested the intermediate samples "from the first time point after the significant change is found", which a change found at 6 months would make 9 months, not the 6 of Table 9. They are now tested at each of their time points from the one at which the change is found.
- The investigation INV-27-031, opened in 2026, is now INV-26-031.

### Aspects

The blind annotation has exactly the paragraphs of the reference answers for each of the four aspects, and leaves out the same ones.

The header rows of tables are linked context of the aspects whose rows they label (`complex_stability.neutral.json`): a header row has no result of its own, but it gives the rows their columns, i.e. their time points, and a model that extracts it with the rows is not wrong. The other cases have no such entries. So are the headings of the sections of the batches, which start with the caption of a table, so that their section holds no paragraph and they would otherwise be scored. The borderline and linked paragraphs that the blind annotation was unsure about are listed with the reasons in the same file: e.g. 3.2, the rule that explains why the intermediate results of V20-2604 start at 6 months, and 4.2.1 and 4.3.1, the statements of the narrative that the results outside the specification contradict.

### After the first runs

The runs of `stability_2026-10-20_text` (`openai/gpt-6-luna` and `openai/gpt-5.6-luna`, 3 runs each, and the three hosted decision models) were compared with the reference answers, as for the other cases. Rulings:

- **Shelf-life acceptance criteria**: 2.1 [12], which 4 of the 6 LLM runs leave out, is now borderline: it says that the stability samples are tested against the shelf-life specification, but also introduces the release specification, which the aspect leaves out (the blind annotation was unsure about it too). 3.3 [38], which all the decision models add, is linked context: the definition of a significant change applies the shelf-life specification to results. 7.3 [116], which they add too, is borderline: a rule on results outside the specification, but a commitment.
- **Stability results of V20-2604**: 5.5 [101], which all the decision models leave out, is now borderline: it closes the investigation of the batch, and says that no other result of the study was outside the specification.
- **Results outside the shelf-life specification**: [38] is linked context and [116] borderline here too. The 6-month impurity D of V20-2607 [91], which 5 of the 6 LLM runs leave out, stays: it is the result outside the specification that only the tables give, i.e. what the case tests.
- What else the decision models add or leave out on the text are table rows, which they cannot tell apart without the caption and the header of their table, and rows of results within the specification: these stay as they are.

The judge agrees with 75 of the 76 checks (`judge --checks`): of the 14 checks of this case, it judges correct the wrong justification that 6.2 applies because the intermediate samples of V20-2604 showed a significant change (a probability of 0.55), where 6.2 requires that they show none.

## complex_multilingual (2026-10-27)

### How the case was made

An AI agent wrote the document and its reference answers together, on 2026-10-27, to measure whether the LLMs write their output in the language that `DocumentLLM.output_language` asks for when the document is in several languages, and whether the extraction holds against `"en"`. The document is a fictional exclusive distribution agreement between a German pump manufacturer and a French distributor; the parties, the places (Lindenfurt, Bellerive-sur-Erdre) and the figures are made up, and the registration number of the distributor fails the check digit of French company numbers.

| Part | Language | Paragraphs |
|------|----------|------------|
| The agreement (Vertriebsvertrag), §§ 1 to 9 and the signatures | German | 0 to 48 (49) |
| Annex 1: the Contract Products, their technical data and list prices in two tables, the payment and delivery terms | English | 49 to 68 (20) |
| Amendment No. 1 (Avenant n° 1) | French | 69 to 79 (11) |

German is 61 % of the paragraphs, 60 % of the characters and 55 % of the words. No paragraph is in Japanese, Chinese or Portuguese, so every target language but English needs translation. The gold file gives the first and the last paragraph of each language (`paragraph_languages` in its `custom_data`).

The answers depend on how the languages interact, and 9 of the 16 need two of them:

- **The French amendment changes German clauses**: it adds Belgium to the exclusive territory and allows active sales in Luxembourg without exclusivity (2.1 to 2.3), cuts the minimum purchase quantity of 2028 by 25 % (3.1), and moves the deliveries from 1 March 2028 from FCA Lindenfurt to DAP at the warehouse of the distributor, with the passing of risk (5.1, 5.2 and point 3.1 of Annex 1).
- **The English annex states three terms otherwise than the German text**, and 9.3 decides between them: the German text prevails, except for technical data and list prices, for which Annex 1 prevails, even though Annex 1 claims precedence for itself. So the payment term is the 30 days of 4.6, not the 45 days of Annex 1; the list price of HP-350 is the EUR 7,250 of Table 1.1, not the EUR 7,150 of 4.2; and the guaranteed efficiency of HP-500 is the 76 % of Table 1.1, not the 78 % of 6.4.
- **Defined terms cross the languages**: Vertragsprodukte, Contract Products and Produits Contractuels; Vertragsgebiet, Territory and Territoire; Verschleißteile and Wear Parts, whose warranty period 6.3 sets and which Annex 1 lists.
- **Distractors**: the HP-700 series, which Annex 1 lists with a price for information only and which the amendment only plans to add by a separate amendment; Luxembourg, open to the distributor without exclusivity.

The 16 concepts: 9 typed ones (dates, numbers, a Boolean, labels), whose values do not depend on the output language; 6 string concepts with short reference values in English, each with one accepted wording; and one JSON object concept with one text field (`description`), whose objects are identified by `series` (`object_key`). Every concept and every aspect asks for justifications, which are written in the output language, and every concept for references.

### Computed answers

Recomputed by the script that wrote the reference answers (not kept); each is also in the `custom_data` of its concept:

- **Net value of the initial stocking order**: 10 × 4,850 + 6 × 7,250 + 2 × 11,400 = EUR 114,800, less the introductory discount of 8 % (4.3): EUR 105,616. With the EUR 7,150 of 4.2 it would be EUR 105,064, and without the discount EUR 114,800.
- **Payment due date of its invoice**: invoiced on Monday 15 March 2027 (Annex 1, 2.1), a day that 4.6 does not count; 30 days end on Wednesday 14 April 2027. The 45 days of Annex 1 would end on 29 April 2027.
- **Minimum purchase quantity for 2028**: 120 × 0.75 = 90 pumps.
- **Effective date of Amendment No. 1**: the date of the last signature (Article 7), 19 November 2027, after that of the distributor on 12 November 2027.
- **Incoterm of a delivery on Wednesday 12 April 2028**: a delivery from 1 March 2028, so DAP.

### Paragraphs not scored

The 17 headings, the captions of the two tables among them. The header row of Table 1.1 is linked to the aspects of the prices and of the warranty, whose rows it labels. The borderline and linked paragraphs of each aspect are listed with their reasons in `complex_multilingual.neutral.json`, e.g. the HP-700 series (1.3 of Annex 1 and Table 1.2), which is not a Contract Product, for the prices and the warranty, and the clause of the agreement by which an amendment prevails (9.4), for the territory and the delivery.

### Points for the review

- **The guaranteed efficiency of HP-500**: 6.4 states 78 % as the value of Annex 1, which gives 76 % as a guaranteed minimum, and 9.3 makes Annex 1 prevail for technical data. A reader could take the 78 % as a separate, stricter guarantee of 6.4, which a minimum of 76 % does not contradict.
- **The payment term**: the exception of 9.3 covers technical data and list prices only, so the 30 days of the German text prevail; a reader who counts payment terms as prices would take the 45 days of Annex 1.
- **Two string answers are longer than the others**: the passing of the transport risk, before and after 1 March 2028, and the changes made by Amendment No. 1, three changes in one sentence.

### Blind review

On 2026-10-27, a separate agent that saw only the numbered text and the definitions of the aspects and concepts derived every answer, selected the paragraphs of every aspect, and checked the dates, the weekdays and the arithmetic.

- **Concepts: all 16 agree.** The 10 typed answers and the objects of the JSON object concept are those of the reference answers, with the same clauses deciding: 9.3 for the price of HP-350, the efficiency of HP-500 and the payment term, and Article 7 of the amendment for its effective date. Its 5 string answers state the same as the reference values, some with a detail more, e.g. that the amendment is subject to the same law and court. It read the two points for the review as the reference answers do: 76 %, as 6.4 derives its figure from Annex 1 ("Danach"), and 30 days, as a payment term is neither technical data nor a list price.
- **No error in the document.** The conflicts that it lists are those written into the case (4.2 and Table 1.1, 6.4 and Table 1.1, 4.6 and point 2.2 of Annex 1, Annex 1 claiming precedence against 9.3), each resolved by 9.3 as the reference answers do. It also noted that the early payment discount of 4.6 does not contradict "no further discounts" of 4.3, a discount for early payment not being a price reduction, and that 7.2 keeps a termination for good cause, which a shortfall alone does not give (3.3). The weekdays and the cross-references all check out.
- **Aspects**: its paragraphs are those of the reference answers, with these differences, ruled against the text:
  - Territory: it adds 11 (2.4, the non-compete), already borderline.
  - Prices and Warranty: it adds 56, the header row of Table 1.1, already linked; and to the warranty, 34 (6.7, liability) as leaning to include, already borderline.
  - Delivery: it leans to include 64 (the handover of the initial stocking order at the Lindenfurt plant on 15 March 2027) and 68 (spare parts in stock dispatched within 5 working days, a lead time). Both are now borderline for the aspect: 64 gives the place and the day of one delivery under the FCA terms of 5.1 rather than a delivery term, and the second sentence of 68 is a lead time of spare parts in a clause on their supply, while 3.1 of Annex 1 gives the lead times of the pumps.
- **Descriptions it found open**: whether the efficiency "as it applies under the agreement" is that of 6.4 or of Annex 1, and whether the effective date of the amendment is its entry into force or the dates its articles apply from (1 January and 1 March 2028); it chose as the reference answers do (76 %, the entry into force). The descriptions are kept: 9.3 and Article 7 decide them.
