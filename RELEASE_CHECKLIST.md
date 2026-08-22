# Release checklist for v1.0.0

The repository must not be described as the final archived release until every item below is closed.

- [x] Isolated repository directory; the clinical project root is not initialised as Git.
- [x] Five-image preprocessing and model source included.
- [x] Frozen 15-member inference, member-specific temperatures, arithmetic probability mean and direct argmax implemented.
- [x] Post-argmax binary mapping and `q_R` analysis-only role tested.
- [x] Statistical evaluation and quantitative figure/table generation included.
- [x] Synthetic five-image example included; no real participant input included.
- [x] All 15 local checkpoint SHA-256 values verified during the full-ensemble synthetic smoke test.
- [x] Confirm GitHub owner and final repository name: `Cozy-CAIZI/BR-SAT`.
- [x] Confirm the creator citation name: `Weijie Yuan` (ORCID not supplied).
- [x] Confirm the institution-approved software licence: `BSD-3-Clause`.
- [x] Confirm the `v1.0.0` scope: source code public; frozen model weights not public.
- [ ] Document the verified rationale for withholding weights and any access conditions or request route.
- [x] Retain the authenticated GitHub account's current public email for public commits.
- [x] Add standard BSD-3-Clause `LICENSE` and verified `CITATION.cff` metadata.
- [x] Verify that all 15 controlled-source weights match the frozen manifest and are individually below GitHub's 100 MB object limit.
- [x] Keep all model-weight files outside the public repository under the `v1.0.0` release policy.
- [x] Add automated compilation, decision-rule tests, public-boundary audit and tag-only final-release gates.
- [ ] Complete the final secrets/privacy scan and inspect the Git reachable object history after all release files and weights are committed.
- [x] Create the public GitHub repository and merge the source-code candidate to `main`.
- [ ] Create the `v1.0.0` tag and GitHub Release.
- [ ] Archive the exact source-code release in Zenodo and test the public landing page.
- [ ] Replace manuscript placeholders only with the resolving repository record, DOI/PID and approved licence.
