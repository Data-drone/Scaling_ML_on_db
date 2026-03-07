# Configs Folder Guide

This folder contains human-maintained experiment track configuration and notes.

## Scope

- These files are reference/config artifacts for planning and repeatability.
- The active Databricks Asset Bundle deployment source is `databricks.yml`.
- Notebooks and scripts in this repo do not auto-load these YAML files today.

## Files

- `single_node.yml` - Single-node CPU baselines and pending runs.
- `ray_scaling.yml` - Ray distributed scaling matrix and OMP guidance.
- `ray_plasma.yml` - Ray object-store tuning experiments.
- `gpu_scaling.yml` - GPU scaling matrix and open questions.

## Maintenance Rules

- Keep experiment definitions and results, but avoid line-number references to docs.
- Prefer explicit status fields (`active`, `planned`, `completed`) over free-form comments.
- Treat these files as planning/record artifacts unless code begins consuming them directly.
