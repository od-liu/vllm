# Real-Time Output Fix for Parallel Sweep

## Problem

When running `run_parallel_sweep.py`, the subprocess output was buffered, causing:
1. **No visible progress**: User couldn't see what was happening
2. **Appeared frozen**: Process looked stuck even though it was running
3. **Difficult debugging**: No way to see where execution was stuck

## Solution

### Changes Made

#### 1. Switched from `subprocess.run()` to `subprocess.Popen()`

**Before:**
```python
result = subprocess.run(
    cmd,
    env=env,
    capture_output=True,  # ❌ Buffers all output until completion
    text=True,
    timeout=3600,
)
```

**After:**
```python
process = subprocess.Popen(
    cmd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # ✅ Merge stderr into stdout
    text=True,
    bufsize=1,  # ✅ Line buffered
    universal_newlines=True,
)

# Stream output line by line
for line in process.stdout:
    print(line, end='')  # ✅ Print immediately
```

#### 2. Added Python `-u` Flag (Unbuffered Mode)

**Before:**
```python
cmd = [
    sys.executable,
    "benchmarks/benchmark_operators.py",
    "--config", config_path,
]
```

**After:**
```python
cmd = [
    sys.executable,
    "-u",  # ✅ Unbuffered output
    "benchmarks/benchmark_operators.py",
    "--config", config_path,
]
```

## How It Works

### Output Flow

```
benchmark_operators.py
  ↓ (with -u flag, no buffering)
stdout/stderr
  ↓ (merged into stdout)
Popen pipe
  ↓ (line buffered, bufsize=1)
for loop
  ↓ (immediate print)
Terminal (real-time display)
```

### Key Components

1. **`-u` flag**: Forces Python to use unbuffered mode for stdout/stderr
2. **`bufsize=1`**: Line-buffered mode in Popen
3. **`stderr=subprocess.STDOUT`**: Merges error messages into main output stream
4. **Immediate `print()`**: Displays each line as soon as it's received

## Benefits

### Before Fix
```
$ python benchmarks/run_parallel_sweep.py ...
Running command: ...
CUDA_VISIBLE_DEVICES: 3,4,5,6

[10 minutes of silence... appears frozen]
```

### After Fix
```
$ python benchmarks/run_parallel_sweep.py ...
Running command: ...
CUDA_VISIBLE_DEVICES: 3,4,5,6
--------------------------------------------------------------------------------
Live output from benchmark:
--------------------------------------------------------------------------------
[2025-11-24 02:00:00] INFO: Initializing LLM...
[2025-11-24 02:00:05] INFO: Loading model...
[2025-11-24 02:00:15] INFO: Model loaded successfully
[2025-11-24 02:00:20] INFO: Starting benchmark...
[2025-11-24 02:00:25] INFO: Processing request 1/30...
[2025-11-24 02:00:30] INFO: Processing request 2/30...
...
```

✅ **Live progress visible**  
✅ **Can see where execution is**  
✅ **Easy to identify if stuck**  
✅ **Better user experience**

## Usage

No changes needed! Just run as before:

```bash
python benchmarks/run_parallel_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/test
```

You'll now see live output from each benchmark run.

## Debugging Stuck Processes

If you see the process is truly stuck (no new output for >5 minutes), you can:

### 1. Check Process Status

```bash
# In another terminal
ps aux | grep benchmark_operators
```

### 2. Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### 3. Check Python Processes

```bash
ps aux | grep python | grep vllm
```

### 4. Kill Stuck Process

```bash
# Find the PID from ps output
kill -9 <PID>

# Or kill all vllm processes (careful!)
pkill -f benchmark_operators
```

## Common Causes of Stuck Processes

### 1. Model Download

**Symptom**: Stuck at "Loading model..." for a long time  
**Cause**: Model being downloaded from HuggingFace  
**Solution**: Pre-download model or check network connection

### 2. OOM (Out of Memory)

**Symptom**: Process appears to hang after "Initializing LLM"  
**Cause**: Insufficient GPU memory  
**Solution**: Reduce `gpu_memory_utilization` in config (e.g., 0.3 → 0.15)

### 3. NCCL Initialization

**Symptom**: Stuck at "Initializing distributed environment"  
**Cause**: NCCL communication issues in multi-GPU setup  
**Solution**: Check network, NCCL_DEBUG=INFO for more info

### 4. Deadlock in Multi-Process

**Symptom**: DP>1 config stuck without errors  
**Cause**: Process coordination issue  
**Solution**: Check logs for "Starting independent process X/Y"

## Testing

Test that real-time output works:

```bash
# Simple test
python3 -c "
import subprocess
import sys

cmd = [sys.executable, '-u', '-c', '''
import time
for i in range(10):
    print(f\"Progress: {i}/10\")
    time.sleep(1)
''']

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    print(line, end='')

process.wait()
print(\"Done!\")
"
```

You should see output every second, not all at once at the end.

## Technical Details

### Why `-u` is Important

Python's default output buffering behavior:
- **With buffering** (default): Output accumulated until buffer full or program exits
- **Without buffering** (`-u`): Output written immediately

### Why `bufsize=1` is Important

Popen buffering modes:
- `bufsize=0`: Unbuffered (not available for text mode)
- `bufsize=1`: Line buffered (output after each newline)
- `bufsize>1`: Block buffered (output after N bytes)
- `bufsize=-1`: System default (usually block buffered)

### Why Merge stderr into stdout

```python
stderr=subprocess.STDOUT  # Merge stderr into stdout
```

Benefits:
- Single stream to monitor
- Maintains chronological order of messages
- Simpler code (one loop instead of two)

## Related Files

- `benchmarks/run_parallel_sweep.py`: Main script (modified)
- `benchmarks/benchmark_operators.py`: Subprocess being executed
- `benchmarks/PARALLEL_SWEEP_GUIDE.md`: Usage guide

## Summary

✅ **Real-time output**: See progress as it happens  
✅ **Better debugging**: Identify where process is stuck  
✅ **Same interface**: No user-facing API changes  
✅ **More confidence**: Know the process is actually running  

This fix significantly improves the user experience when running long-running benchmark sweeps.


