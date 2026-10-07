# 09 — Vision (stretch)
**Capability:** a vision-capable / `flux` model.
**Build:** feed an error-screenshot image, extract the error text, and route it to the right
intent. Use a handful of sample images you generate or screenshot.
**Definition of done:** `results.json` (extraction accuracy on the samples); `NOTES.md`.
Check the catalog first — needs a multimodal model; if none is available, document that.

## Reproduce
No vision model is serverless here; `qwen3-vl-8b-instruct` runs on-demand (see NOTES).
```bash
python generate_images.py     # render 5 synthetic error screenshots + labels.jsonl

firectl create deployment accounts/fireworks/models/qwen3-vl-8b-instruct \
  --deployment-shape accounts/fireworks/deploymentShapes/qwen3-vl-8b-instruct-throughput \
  --scale-to-zero-window 20m
# set DEPLOYMENT in run.py; first call cold-starts (retry on 503 DEPLOYMENT_SCALING_UP)
python run.py                 # extract error text + route intent on each image
firectl delete deployment <DEP> --ignore-checks
```

