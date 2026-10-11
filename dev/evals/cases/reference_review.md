# Review of the Reference Answers (2026-10-11)

The reference answers of both cases (`complex_nda.gold.json`, `complex_dpa.gold.json`) were produced by an AI agent. This is the record of their review, which was also done by AI agents: no person has verified the answers. The maintainer delegated the review.

## Method

1. **Blind annotation.** For each case, a separate agent got only the text with numbered paragraphs and the definitions of the aspects and concepts. It saw neither the reference answers nor any model result. It listed the paragraphs of each aspect and derived the answer of each concept.
2. **Own derivation.** The reviewing agent read both contracts in full and derived every typed concept answer from the text before comparing it with the reference.
3. **Comparison of three sources**: the reference, the blind annotation, and the five decision models of the baseline (how many of them give a paragraph a probability of 0.5 or more for an aspect).
4. **Ruling.** Every paragraph on which the reference and the blind annotation differ, and every paragraph on which four or five decision models disagree with both, was ruled on against the text.
5. **Second part**, after the baseline was recorded again with runs that keep what they extract: the evidence of each concept, the justifications and the string answers were reviewed, and what the hosted LLMs extract beyond the reference (seven runs) was ruled on in the same way.

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

The baseline was recorded again after the first part of the review, as the earlier runs had kept their scores but not what they extracted. The runs now keep it, and were scored again from it after the second part (`rescore`). Aspect F1 before the review and after it (NDA / DPA, mean of the runs of a configuration):

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
