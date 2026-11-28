# Fever Dreams v2 Roadmap

**Document Style:** ~70% ASD-STE100.  
**Status:** Planning and design. Build in phase order.  
**Hardware Target:** Apple Silicon Mac, 48 GB unified memory, 100% local AI execution.

## Build Phases

1. **[Phase 1 — Engine, Subsystems, and Eval Harness](01-engine-and-subsystems.md)** *(Complete — see **[Phase 1 Findings](01-phase1-findings.md)**)*  
   Build and test the style-agnostic rendering, continuity, set-reuse, Critic-to-Tooling recovery loop, and eval subsystems first. Prove defect detection, closed-loop repair, and model right-sizing in minutes using real fixtures before running full pipelines.
2. **[Phase 2 — Agentic Creation Pipeline](02-agentic-creation-pipeline.md)**  
   Upgrade the local Writers' Room with multi-model pitching, adversarial story selection, a Continuity Expert, an Art Direction Critic loop, and Story Trap evals.
3. **[Phase 3 — New Stop-Motion Book and Film](03-stop-motion-book-production.md)**  
   Add the 12 fps stop-motion video cadence, video QC, and video-to-book frame extraction, run a 2-page micro-pilot, then produce the next full-length stop-motion Fever Dream cartoon and matching Little Golden Book.

## Cross-Phase Hardware Rule (48 GB Unified Memory)

Do not load two large models at the same time (for example, two 18 GB LLMs, or `mflux` with a 30B VLM). Every phase must **batch by model**:

1. Load Model A -> run all queued tasks for Model A -> unload Model A (`lms unload --all` or Python GC + `mx.clear_cache()`).
2. Load Model B -> run all queued tasks for Model B -> unload Model B.
