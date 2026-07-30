# Research corpus

The long-form public-source briefing is
`Cn2_Advanced_Modeling_Research_Briefing.tex`. It includes the full research
landscape, 32 annotated priority papers, 16 dataset cards, an experiment
backlog, and the 193-source discovery index. The original DOCX remains under
`archive/` for layout fidelity.

Compile the LaTeX source from the repository root:

```bash
uv run tectonic -X compile \
  research/Cn2_Advanced_Modeling_Research_Briefing.tex \
  --outdir reports/generated/research-briefing
```

`CN2_RESEARCH_BRIEF.md`, `ARCHITECTURE.md`, and `datasets.yaml` are working
decision artifacts. Completed experimental evidence is generated from the
LaTeX template under `reports/templates/`.
