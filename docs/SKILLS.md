# Project-local skills

The repository vendors skills under `.agents/skills/` so a clean Codex session
discovers the same operating guidance from the repository root. External skills
were found and installed with the skills.sh CLI in copy mode; `skills-lock.json`
records their upstream source and content hash.

| Skill | Source | Purpose |
| --- | --- | --- |
| `cn2-research-pipeline` | Custom, created with Codex `skill-creator` | Cn² provenance, leakage, 16 GB models, guarded iteration, LaTeX completion |
| `mlflow-onboarding` | `mlflow/skills` | Traditional deep-learning run, artifact, and model tracking |
| `pytorch-lightning` | `davila7/claude-code-templates` | Structured training, callbacks, precision, checkpointing |
| `latex-formatting` | `lingzhi227/agent-research-skills` | LaTeX structure and validation |
| `literature-search-arxiv` | `google-deepmind/science-skills` | Rate-limited paper search/retrieval with terms notice |
| `scientific-brainstorming` | `k-dense-ai/claude-scientific-skills` | Explicit assumptions, adversarial review, decision logs |
| `frontend-design` | `anthropics/skills` | Subject-specific operational UI direction |
| `vercel-react-best-practices` | `vercel-labs/agent-skills` | React fetching, rendering, and bundle discipline |
| `skypilot-multi-cloud-orchestration` | `davila7/claude-code-templates` | Portable, recoverable, cost-aware cloud job specifications |
| `vastai` | `vast-ai/vast-cli` | Provider-native offer, instance, cost, and teardown operations |

## Selection notes

- Official-owner skills were preferred where available (MLflow, Google
  DeepMind, Anthropic, Vercel, Vast.ai).
- The generic Lightning and SkyPilot skills are secondary guidance; current
  official package/docs versions and repository tests control when they differ.
- No skill can override `AGENTS.md`, sealed evaluation gates, dataset terms,
  budget ceilings, or the cloud-approval requirement.
- The vendored copies were reviewed and normalized only at the frontmatter
  level so they pass the repository's current skill validator. Their original
  upstream hashes remain in `skills-lock.json` for comparison.
- The arXiv skill may be used only after its required terms notice is recorded;
  it rate-limits requests and paper licenses must be checked individually.

## Validation

```bash
for skill_dir in .agents/skills/*; do
  python /root/.codex/skills/oai/skill-creator/scripts/quick_validate.py "$skill_dir"
done
```

The custom installed skill is also available directly in Codex:
<https://chatgpt.com/skills?skill_id=6a6a951b58e88191b7d2f77c1111aad2>.
