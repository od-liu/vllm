# E2E Sweep Quick Start Guide

## 🚀 Three Steps to Benchmark

### Step 1: Prepare Config (2 minutes)

Copy and edit the template:
```bash
cd /mnt/disk1/ljm/vllm
cp benchmarks/e2e_sweep/example_configs/e2e_config_template.py my_e2e_config.py
```

Edit `my_e2e_config.py` - **only 3 things required**:
1. `model_path` - your model path
2. `trace_file_path` - your trace file path  
3. `trace_time_range_minutes` - how much trace data to use (e.g., `(0, 2)` for 2 minutes)

### Step 2: Run Sweep

```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_e2e_config.py \
    --output-dir results/my_first_sweep \
    --gpu-type h100
```

**What happens:**
- Generates all TP×DP combinations (e.g., for 8 GPUs: TP=8/DP=1, TP=4/DP=2, TP=2/DP=4, TP=1/DP=8)
- Runs E2E benchmark for each configuration
- Saves results to `results/my_first_sweep/`
- Auto-generates visualizations

**Time estimate:** ~10-30 minutes depending on:
- Number of GPUs (more GPUs = more configs)
- Trace length (longer trace = slower)
- Model size

### Step 3: View Results

Check the plots:
```bash
ls results/my_first_sweep/plots/
```

**Generated files:**
- `e2e_comparison.png` - Bar charts of all metrics
- `scaling_analysis.png` - How metrics scale with GPU count
- `efficiency_analysis.png` - Throughput per GPU
- `percentiles_distribution.png` - P50/P90/P99 distributions
- `e2e_summary.txt` - Text summary

**View the summary CSV:**
```bash
cat results/my_first_sweep/e2e_summary.csv
```

---

## 📋 Common Use Cases

### Use Case 1: Quick Test (4 GPUs)
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config my_e2e_config.py \
    --output-dir results/test_4gpu \
    --gpu-ids "0,1,2,3"
```

### Use Case 2: Production Scale (8 GPUs, 5min trace)
Edit config: `trace_time_range_minutes=(0, 5)`
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_e2e_config.py \
    --output-dir results/prod_8gpu \
    --gpu-type h100
```

### Use Case 3: Resume Interrupted Sweep
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_e2e_config.py \
    --output-dir results/my_sweep \
    --skip-existing
```

---

## 🎯 Understanding Your Results

### Key Metrics

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **TTFT** | Time to first token | Lower is better (< 100ms ideal) |
| **TPOT** | Time per output token | Lower is better (< 50ms ideal) |
| **Throughput** | Tokens per second | Higher is better |
| **E2E Latency** | Total request time | Lower is better |

### Interpreting Configurations

**TP=8, DP=1** (High Tensor Parallel):
- ✅ Good for: Large models, low latency
- ❌ Bad for: High throughput

**TP=1, DP=8** (High Data Parallel):  
- ✅ Good for: High throughput, small models
- ❌ Bad for: Large models that don't fit in 1 GPU

**TP=4, DP=2** (Balanced):
- ✅ Good for: Medium models, balanced workload

### Which Config to Choose?

Look at the plots in `results/*/plots/`:

1. **e2e_comparison.png**: 
   - Which config has lowest TTFT? → Best for latency-sensitive apps
   - Which config has highest throughput? → Best for batch processing

2. **efficiency_analysis.png**:
   - Which config has highest throughput per GPU? → Most efficient

3. **dp_vs_tp_*.png** (if available):
   - Direct comparison of strategies for same GPU count

---

## ⚠️ Troubleshooting

### Error: "Out of memory"
**Solution:** Edit config, reduce `gpu_memory_utilization`:
```python
gpu_memory_utilization=0.3,  # Try lower value
```

### Error: "No trace requests loaded"  
**Solution:** Check trace file path and time range:
```python
trace_file_path="/correct/path/to/trace.jsonl",
trace_time_range_minutes=(0, 2),  # Make sure this has data
```

### Benchmark is too slow
**Solution:** Reduce trace time range:
```python
trace_time_range_minutes=(0, 1),  # Use only 1 minute
```

---

## 📖 Full Documentation

For detailed information, see [README.md](README.md)

For configuration options, see [example_configs/e2e_config_template.py](example_configs/e2e_config_template.py)

---

## 💡 Pro Tips

1. **Start small**: Test with 2-minute trace first, then scale up
2. **Use skip-existing**: Resume failed sweeps with `--skip-existing`
3. **Compare apples to apples**: Use same trace data for all experiments
4. **Check text summary**: If plots fail, read `plots/e2e_summary.txt`
5. **Document GPU type**: Always use `--gpu-type` flag for your records


