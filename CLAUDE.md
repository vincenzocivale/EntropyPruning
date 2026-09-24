# EAF contributor notes

Use the Conda environment in `environment.yml` and run commands from the repository root.
`$EAF_WSI_ROOT` is the single runtime root for data, caches, checkpoints, results and logs.

The primary EAF training corpus is unlabeled HISTAI. THUNDER and TCGA/CPTAC
cohorts are downstream evaluation only. The final WSI experiment must keep the pruned
student source cache separate from the full teacher cache.

Supported workflow: `README.md` and `docs/pipeline.md`.
