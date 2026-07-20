# arXiv submission prep

**SUBMISSION AWAITS BAMDAD'S EXPLICIT GO — do not submit.**

This file is preparation only. Nothing here has been submitted to arXiv or emailed
to anyone. It exists so that, once Bamdad says go, the mechanical steps are ready.

---

## Part 1 — cs.LG endorsement request (email draft)

> **To:** (an arXiv author already able to submit to cs.LG)
> **Subject:** Endorsement request for arXiv cs.LG submission
>
> Dear Dr. ___,
>
> I am an independent researcher preparing a short note for arXiv and am writing to
> ask whether you would be willing to endorse me to submit to cs.LG. The note is an
> honest-negative reproduction of the concept-injection introspection protocol of
> Lindsey et al. (2025) on open Qwen2.5 models: across a base / general-instruct /
> code-instruct grid at 7B, 14B, and 32B, every rung returns a null on strict
> correct-identification except one marginal cell (Qwen2.5-Coder-32B, 5/216), which
> does not replicate down its own size ladder, so I report the effect as a
> conjunction of code-heavy post-training and roughly 32B scale rather than a main
> effect of either. The note also documents a dose-fragility result: an
> injection strength calibrated for coherent steering sits well below the source
> paper's regime and produces a clean but false null. Code, raw transcripts, and the
> dose calibration are public. I would be glad to send the PDF if it is useful for
> your decision.
>
> Thank you for considering it.
>
> Kind regards,
> Bamdad Dashtban

Notes for Bamdad:
- If arXiv assigns an endorsement code, paste it into the endorsement page; the
  endorser does not need to read the paper unless they want to.
- Keep the wording factual and neutral; do not oversell. The note is a null result.

---

## Part 2 — submission checklist

- [ ] **Primary category:** cs.LG
- [ ] **Secondary category:** cs.CL
- [ ] **License:** recommend arXiv non-exclusive license to distribute, with
      **CC BY 4.0** — *Bamdad's call*; change if you prefer a more restrictive
      license.
- [ ] **Author metadata:** name **Bamdad Dashtban**; affiliation **Independent**
      (unless Bamdad wants a different affiliation).
- [ ] **Title:**
      *Concept-Injection Introspection in Open Models Is Dose-Fragile; Its One
      Above-Chance Signal Does Not Replicate Across Scale*
- [ ] **Abstract (paste-ready):**

  We reproduce the concept-injection introspection protocol of Lindsey et al.
  (2025, "Emergent Introspective Awareness in Large Language Models") on open
  Qwen2.5 models. Two findings result. First, the effect is dose-fragile: an
  injection strength calibrated for coherent activation steering sits roughly 4 to
  18 times below the paper's absolute strength, and at that under-dose every model
  returns a clean null that passes every sanity check (flat controls, a working
  positive control, coherent transcripts). Correcting the dose to the paper's regime
  and adding a fails-loud judge is what surfaces any signal at all. Second, at the
  corrected dose the picture is a null across the board with a single exception that
  does not generalize. Filling a base / general-instruct / code-instruct grid at 7B,
  14B, and 32B, every rung scores 0/216 strict correct-identification except
  Qwen2.5-Coder-32B, which scores 2.3% (5/216), above both a no-injection and a
  random-direction control with non-overlapping 95% CIs. That one above-chance cell
  does not replicate down the Coder size ladder: Coder-7B is 0/216 and Coder-14B is
  1/216 strict (two trials named the concept, one incoherent, so one passes the
  strict coherent-and-correct rule), both with CIs overlapping the 0.000 controls.
  The honest reading is a conjunction — the signal appears only where code-heavy
  post-training meets ~32B scale — not a fine-tune main effect and not a scale main
  effect, and it rests on one marginal cell. Within the 32B row the base control
  still rules out parameter count (all three are 32B) and fine-tuning in general
  (Instruct is a fine-tune and is null), and a logit-lens localizes that cell's
  mechanism: the injected concept is linearly decodable at the unembedding in
  Coder-32B but not in base or Instruct, a legibility difference introduced by
  code-heavy post-training rather than suppression. We report the effect with the
  caveat that it is small, rests on a single a-priori dose, and does not survive its
  own size ladder. An earlier version of this note framed the 32B result as
  fine-tune-dependent rather than scale-dependent; the size ladder retracts that,
  since within the Coder family the effect is present only at 32B.

- [ ] **Figure included:** `results/scaling_trend_k2.png` is embedded in the PDF
      (Results section). For an arXiv source upload, include `note.tex` plus the
      figure file, keeping the relative path `../results/scaling_trend_k2.png`, or
      flatten the paths and bundle the PNG alongside `note.tex`.
- [ ] **Files to upload:** either `note.pdf` (PDF-only submission) or the LaTeX
      source (`note.tex` + the PNG). PDF-only is simplest for a single-file note.
- [ ] **Comments field (optional):** note the companion write-up and the public
      code/transcripts repository if desired.

**SUBMISSION AWAITS BAMDAD'S EXPLICIT GO — do not submit.**
