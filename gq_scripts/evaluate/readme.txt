这几个常用评估脚本的作用如下：

`run_inference_and_eval.sh`
- 标准 image model 一键推理 + 评估脚本。
- 先调用 `batch_inference.py` 逐切片做分割，再调用 `batch_evaluate.py` 计算 3D 指标。
- 适合做纯 image model baseline。

`run_context_inference_and_eval.sh`
- slice-context v1 的一键推理 + 评估脚本。
- 先调用 `batch_inference_context.py` 做带切片上下文的推理，再复用 `batch_evaluate.py` 做 3D 评估。
- 适合和普通 image model baseline 做对比。

`run_tracker_auto_seed_inference_and_eval.sh`
- image model + tracker 的自动测试脚本。
- 先调用 `tracker_auto_seed_inference.py` 做自动 seed / hybrid 推理，再调用 `evaluate_tracker_coco_predictions.py` 做 3D 评估。
- 这条链 inference 时不看 GT seed，evaluate 时会读取 GT 来计算指标。

`tracker_auto_seed_inference.py`
- `run_tracker_auto_seed_inference_and_eval.sh` 背后的核心 Python 推理脚本。
- 当前实现是 `detector-first + tracker-refine`：
  先让 image model 跑完整个 volume，选出高置信 conditioning frames，再交给 tracker 传播，最终采用“detector 优先、tracker 补空”的混合输出。
