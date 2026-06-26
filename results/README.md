# Submission outputs

Generated files:

- `results/results.json`: required prediction file.
- `results/submission.zip`: flat zip containing only `results.json` at the archive root.

Validate before submitting:

```powershell
python -m src.submit.validate_results --input results/results.json --strict
python -m src.submit.zip_submission --results results/results.json --output results/submission.zip
```

