# Controlled-source to public-release mapping

The Python source files below are byte-identical copies of controlled project source files. Paths are relative to the AIRBPTSAT project root; no local absolute path is retained in the public candidate. The JSON configuration preserves the same parsed content but uses LF line endings in the public repository.

| Public candidate file | Controlled source | SHA-256 |
|---|---|---|
| `src/multimodal_model_nvidia.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/multimodal_model_nvidia.py` | `17775252ca7abe5fe58e8e44c3ae3a77803b29ed9d982ffa14b1d3f5e33ec7ac` |
| `src/multimodal_data_nvidia.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/multimodal_data_nvidia.py` | `f90924a111c489fa9ddd57400ac0761ca1bcd4b81dd9dae46fcc0d3266782d0a` |
| `src/multimodal_cv_model.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/multimodal_cv_model.py` | `500bb3cb11bedb2730ad2370e46280ce865a8aacb401a59c9e0d79937dba7069` |
| `src/multimodal_cv_data.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/multimodal_cv_data.py` | `fc6e1d7dff27f96202755e8b8c319e2fc1aae4acc6ef48be78d3d259f7028a44` |
| `src/clinical_review.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/clinical_review.py` | `938bbda63ce7538ecdd74d4076e9f4939c4bcddd69562b39d994eaff2f23ab99` |
| `scripts/train_multimodal_cv5.py` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/source/code_v2/train_multimodal_cv5.py` | `a6a46aa7e2c6e0d35d2f37a82608deaa22457d9d5b66e787ac35f20fb579efb9` |
| `scripts/plot_figure3.py` | `outputs/BR_SAT_IDP_PANELWISE_REBUILD_20260819/WORK/plot_fig3_verified.py` | `46c73a79342571cd723d0a0f58b084dfb8cace5cbdc1613f3f7bffc7e36cc779` |
| `config/frozen_ensemble_v1.0.0.json` | `outputs/AIRBPTSAT_V2_FROZEN_101_VALIDATION_20260717/provenance/V2_NESTED_EVALUATION_MANIFEST.json` | controlled source `c75f8f4e47ad9045ec7df4200e9fe25beafd4b65e58d75070c5eef3cf1370581`; public LF copy `d95027d009ad31f54ebd4576c58ea704a2a2bca8765a3cfef634c43be8192e83` |

The privacy-minimised inference entry point, decision-rule module, evaluation script, synthetic examples, documentation and repository audit were newly added for the public release candidate. They do not alter the frozen checkpoint contents or the reported primary decision rule.
