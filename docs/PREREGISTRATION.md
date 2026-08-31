# PX-057 Certified Stopping v2 — Confirmatory Preregistration Draft

Date: 2026-08-31
Status: design draft; **not frozen and not yet confirmatory**

## Scientific question

Can a frozen round-level stopping policy receive a finite-sample certificate on
the correctness harm it causes, while paired inference separately determines
whether continued rolling self-revision is net beneficial or itself
inadmissible?

The contribution is the inference procedure and protocol diagnosis. Answer
stability is a published baseline, not a novel stopping method.

## Protocol

The protocol is rolling self-revision: every round receives the original
problem and only the immediately preceding complete solution. It does **not**
receive an accumulated transcript. The confirmatory run must record input,
generated, and total billed tokens for every round.

## Frozen policy candidate

Stop at the first round at or after round 2 for which the normalized extracted
answer is unchanged across two consecutive rounds. Otherwise use round 8. No
confidence threshold is included because it changed no decision in the PX-057
discovery sample.

This policy becomes frozen only when its config, code commit, extraction rules,
dataset manifest, model revision, prompt text, decoding settings, and analysis
plan are timestamped before confirmatory outcomes are inspected.

## Primary estimands and hypotheses

### H1 — Stopping-harm certificate

Estimate `Pr(full correct, stopped wrong)` on one declared exchangeable
model–dataset–prompt distribution. Pass when its one-sided exact 95% upper
confidence bound is at most 0.02. The point estimate cannot pass this gate by
itself.

### H2 — Paired net-effect certificate

Estimate `accuracy(stopped) - accuracy(full)` from paired item outcomes. Pass
when its one-sided exact 95% lower confidence bound is at least -0.01.

### H3 — Dual certificate

Require H1 and H2 simultaneously with a shared 0.05 error budget, split equally
by Bonferroni. Report the component certificates even when the dual gate fails.

### H4 — Protocol admissibility

Compare the preregistered reference round 2 with round 8 on paired items. The
rolling protocol is inadmissible past round 2 when the one-sided lower bound on
`accuracy(round 2) - accuracy(round 8)` exceeds 0.05.

This identifies protocol degradation but is not by itself causal evidence that
self-revision caused it.

### H5 — Causal protocol split

Add a matched context-reset arm of eight independent generations. Each
generation receives the original question and the same first-round instruction,
with no preceding solution. Match model revision, decoding, maximum output,
items, and generation count.

- If reset performance remains within 0.03 of the preregistered round-2
  reference while rolling round 8 degrades, interpret the damage as dependent
  rolling revision.
- If reset performance degrades similarly, reject that interpretation and
  investigate repeated-generation, scoring, or position effects.

Because the original protocol never accumulated the full transcript, this is
not called a context-length ablation.

### H6 — Extraction validity

Run strict-final-answer, boxed-answer, and fallback-last-number extractors before
unblinding arm-level outcomes. All disagreements and empty parses receive
blinded manual adjudication. Report accuracy under every extractor and the
adjudicated primary scorer. If conclusions change across admissible extractors,
the primary scientific claim fails robustness.

## Sample and stopping rule

Target at least 400 fresh confirmatory items. The final size must be selected by
a prospective operating-characteristic calculation, not by the archived single
harm event alone. After the initial 400, apply only the preregistered blinded
sample-size/abandonment rule. If the required total exceeds 2,500, stop and
report that the 0.02 certificate is infeasible for this policy and distribution.

No item used to select prompts, thresholds, extractors, or the reference round
may enter the confirmatory set.

## Cost outcomes

Primary cost reporting uses total billed tokens: input/prefill plus generated
tokens for every request actually made. Generated-token savings are secondary.
Also report request count, latency on fixed hardware, and any provider-specific
cached-input accounting. The archived PX-057 data can support generated-token
savings only.

## Multiplicity and exploratory analyses

The primary policy has `K=1`. Any threshold, patience, minimum-round, extractor,
or prompt search is discovery-only and cannot share the confirmatory set.
Exploratory grid-searched policies must be reported separately with their full
multiplicity correction; they cannot replace the frozen policy after outcomes
are observed.

## Required outputs

- Immutable run manifest and hashes.
- Raw prompts and responses for every request.
- Item-level extractor outputs and blinded adjudications.
- Paired 2×2 tables for stopped versus full and round 2 versus round 8.
- Stopping-harm, paired-net-effect, and dual certificates.
- Rolling-versus-reset causal comparison.
- Billed-token, generated-token, request-count, and latency accounting.
- A verifier that recomputes all metrics from raw generations rather than
  accepting self-reported gate decisions.

## Discovery replay boundary

The archived 200-item PX-057 run is used only to validate code, recover paired
tables, diagnose extraction fragility, and plan the confirmatory experiment. It
cannot certify the successor because the policy, round-2 comparison, and
inference plan were informed by those outcomes.
