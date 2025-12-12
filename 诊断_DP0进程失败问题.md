# 诊断：DP0进程失败导致Benchmark无法完成

## 问题现象

运行完整trace文件（blksz_16.jsonl，6852个请求）时：
- ✅ DP rank 1 成功：3426/3426 requests
- ❌ DP rank 0 失败：输出空文件（0字节）
- ❌ 整体benchmark失败：`RuntimeError: One or more DP ranks failed`
- ❌ 没有生成最终的结果JSON文件

对比使用稀疏文件（k10.jsonl，686个请求）时：
- ✅ 所有DP ranks成功
- ✅ 生成完整的结果JSON

## 根本原因分析

### 1. **DP0进程确实启动了**

日志中可以看到DP0的EngineCore存在：
```
(EngineCore_DP0 pid=1925812) [DP1] WARNING 12-04 08:13:35 block_pool.py:393] 
74 blocks were not freed (ref_cnt > 0). This may indicate a reference counting issue.
```

### 2. **DP0遇到内存问题**

警告信息表明：
- KV Cache blocks没有正确释放
- 引用计数（ref_cnt）大于0
- 可能导致内存泄漏

### 3. **DP0的日志没有显示**

只能看到DP1的日志输出：
```
[DP1] INFO Request 6573: timeout=...
[DP1] WARNING Request 6575 is behind schedule
```

完全看不到DP0的进度信息（应该有类似`[DP0] INFO Request XXX`的日志）

### 4. **处理大量请求时问题加剧**

| Trace文件 | 总请求数 | DP0请求数 | DP1请求数 | 结果 |
|----------|---------|----------|----------|------|
| k10.jsonl | 686 | 343 | 343 | ✅ 成功 |
| blksz_16.jsonl | 6852 | 3426 | 3426 | ❌ DP0失败 |

**结论**：DP0在处理3426个请求过程中遇到致命错误并崩溃。

## 可能的具体原因

### 原因1：GPU内存耗尽（OOM）

**机制**：
- KV Cache blocks泄漏累积
- 处理3426个请求后GPU内存耗尽
- CUDA OOM错误导致进程崩溃

**证据**：
- `74 blocks were not freed` 警告反复出现
- DP0输出文件为空（进程异常退出）

### 原因2：AsyncLLM的并发bug

**机制**：
- DP0和DP1使用相同的GPU（TP=2时共享GPU 4,5）
- AsyncLLM可能没有正确处理多进程共享GPU
- 进程间GPU资源冲突

**证据**：
- 单GPU（DP=1）时成功
- 多进程（DP=2）时DP0失败

### 原因3：trace数据分片问题

**机制**：
- DP0分配的请求有特殊问题（极长输入/输出等）
- 某个特定请求导致EngineCore崩溃

## 已实施的修复

### 修复1：保留失败进程的输出文件

**修改文件**：`/mnt/disk1/ljm/vllm/benchmarks/benchmark_e2e_metrics.py`

**之前（错误）**：
```python
if not all_success:
    # Clean up temp files even on failure
    for file_path in rank_output_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)  # ❌ 删除了！
        except:
            pass
```

**现在（正确）**：
```python
if not all_success:
    # DON'T delete temp files on failure - keep them for debugging
    logger.error("Some DP ranks failed. Temporary output files preserved for debugging:")
    for rank, file_path in enumerate(rank_output_files):
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.error(f"  DP rank {rank}: {file_path} ({file_size} bytes)")
```

### 修复2：保存异常信息到输出文件

**之前**：
```python
except Exception as e:
    logger.error(f"Process {dp_rank} failed: {e}", exc_info=True)
    sys.exit(1)  # 直接退出，无输出
```

**现在**：
```python
except Exception as e:
    logger.error(f"Process {dp_rank} failed: {e}", exc_info=True)
    # Write error info to output file for debugging
    try:
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "dp_rank": dp_rank,
            "gpu_indices": gpu_indices
        }
        with open(output_file, 'w') as f:
            json.dump(error_info, f, indent=2)
    except:
        pass
    sys.exit(1)
```

## 下一步诊断

### 步骤1：重新运行并查看错误详情

```bash
cd /mnt/disk1/ljm/vllm

# 清理Python缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 重新运行benchmark
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/debug_dp0 \
    --gpu-ids "4,5,6,7"
```

### 步骤2：检查DP0的错误文件

```bash
# 查看临时文件目录
ls -lh tmp_benchmark/tp2_dp2/

# 读取DP0的错误信息
cat tmp_benchmark/tp2_dp2/e2e_bench_dp0_*.json
```

**期望看到**：
- DP0的具体错误信息（error字段）
- 完整的stack trace（traceback字段）
- 失败时的GPU状态

### 步骤3：根据错误类型采取措施

#### 如果是OOM错误

**解决方案**：
```python
# 在 e2e_config_template.py 中
gpu_memory_utilization = 0.5  # 从0.8降到0.5
max_model_len = 32768          # 从40960降到32768
```

#### 如果是多进程GPU冲突

**解决方案**：
```bash
# 使用更多GPU，避免共享
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config ... \
    --gpu-ids "4,5,6,7"  # TP=2,DP=2 → 每个进程独立GPU
```

#### 如果是特定请求导致崩溃

**解决方案**：
1. 找出导致崩溃的请求ID
2. 在trace数据中定位该请求
3. 跳过或修复问题请求

## 临时解决方案：使用DP=1

如果问题难以修复，可以先用DP=1模式运行：

```bash
# 单进程，避免多进程问题
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 2 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/h100_8gpu_5min_k1 \
    --gpu-ids "4,5"  # TP=2,DP=1
```

**优点**：
- ✅ 避免多进程同步问题
- ✅ 更容易诊断
- ✅ 日志输出清晰

**缺点**：
- ❌ 无法测试DP性能
- ❌ 处理速度较慢（单进程）

## 关键修复总结

### 修改的文件

1. **`/mnt/disk1/ljm/vllm/benchmarks/benchmark_e2e_metrics.py`**
   - 添加 `import traceback`
   - 保留失败进程的临时文件
   - 将异常信息写入输出文件

### 预期效果

下次运行失败时：
- ✅ 可以查看DP0的具体错误
- ✅ 有完整的stack trace
- ✅ 知道失败的准确原因

## 测试验证

### 清理缓存并重新运行

```bash
cd /mnt/disk1/ljm/vllm

# 1. 清理Python缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 2. 重新运行
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/debug_dp0_failure \
    --gpu-ids "4,5,6,7"
```

### 查看错误信息

```bash
# 失败后查看临时文件
ls -lh tmp_benchmark/tp2_dp2/e2e_bench_dp0_*.json

# 查看最新的DP0错误
cat $(ls -t tmp_benchmark/tp2_dp2/e2e_bench_dp0_*.json | head -1)
```

**期望输出**：
```json
{
  "error": "CUDA out of memory. Tried to allocate ...",
  "traceback": "Traceback (most recent call last):\n  File ...",
  "dp_rank": 0,
  "gpu_indices": "4,5"
}
```

## 结论

DP0进程失败的根本原因尚未完全确定，但已实施修复以保留错误信息。下一步需要：

1. ✅ 重新运行benchmark
2. ✅ 查看DP0的详细错误
3. ✅ 根据具体错误采取针对性修复

修复后的代码会提供更多诊断信息，帮助我们快速定位问题。




