# E2E Benchmark Sweep - File Index

## 📖 Documentation (Start Here!)

1. **[QUICKSTART.md](QUICKSTART.md)** - ⭐ START HERE
   - 3-step quick start guide
   - Common use cases
   - Results interpretation

2. **[README.md](README.md)** - Complete reference
   - Full feature documentation
   - All configuration options
   - Troubleshooting guide
   - Advanced usage

3. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical overview
   - What was implemented
   - Architecture details
   - Comparison with other tools

## 🚀 Scripts

1. **[run_e2e_sweep.py](run_e2e_sweep.py)** - Main sweep script
   ```bash
   python benchmarks/e2e_sweep/run_e2e_sweep.py \
       --num-gpus 8 \
       --base-config my_config.py \
       --output-dir results/my_sweep
   ```

2. **[visualize_e2e_results.py](visualize_e2e_results.py)** - Visualization generator
   ```bash
   python benchmarks/e2e_sweep/visualize_e2e_results.py \
       --results-dir results/my_sweep
   ```

## ⚙️ Configuration

1. **[example_configs/e2e_config_template.py](example_configs/e2e_config_template.py)** - Config template
   - Copy this to create your config
   - Comprehensive inline documentation
   - All parameters explained

## 🎯 Quick Start Command

```bash
# 1. Copy config template
cp benchmarks/e2e_sweep/example_configs/e2e_config_template.py my_config.py

# 2. Edit my_config.py (set model_path, trace_file_path, trace_time_range_minutes)

# 3. Run sweep
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_config.py \
    --output-dir results/my_sweep \
    --gpu-type h100

# 4. View results (auto-generated)
ls results/my_sweep/plots/
```

## 📊 Expected Output

```
results/my_sweep/
├── tp8_pp1_dp1_e2e.json          # Individual config results
├── tp4_pp1_dp2_e2e.json
├── tp2_pp1_dp4_e2e.json
├── tp1_pp1_dp8_e2e.json
├── e2e_summary.csv                # Summary CSV
└── plots/
    ├── e2e_summary.txt            # Text summary
    ├── e2e_comparison.png         # Main comparison
    ├── percentiles_distribution.png
    ├── scaling_analysis.png
    ├── efficiency_analysis.png
    └── dp_vs_tp_8gpus.png
```

## 💡 Recommended Reading Order

1. **First time user?** → Read [QUICKSTART.md](QUICKSTART.md)
2. **Need details?** → Read [README.md](README.md)
3. **Configuring?** → Read [example_configs/e2e_config_template.py](example_configs/e2e_config_template.py)
4. **Technical overview?** → Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

**Ready to benchmark!** 🚀
