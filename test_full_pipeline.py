#!/usr/bin/env python3
"""
完整测试整个pipeline：模拟benchmark_e2e_metrics.py的流程
"""

import asyncio
import sys
sys.path.insert(0, '/mnt/disk1/ljm/vllm')

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.inputs import TokensPrompt
from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceLoader
from vllm.profiler.trace_scheduler import TraceScheduler

async def test_full_pipeline():
    print("=" * 80)
    print("完整Pipeline测试 - 模拟benchmark流程")
    print("=" * 80)
    
    try:
        # 1. 初始化AsyncLLM（和benchmark一样）
        print("\n1. 初始化AsyncLLM...")
        engine_args = AsyncEngineArgs(
            model="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.5,
            dtype="auto",
            enforce_eager=True,
            disable_log_stats=False,
        )
        
        if engine_args.max_model_len is None:
            engine_args.max_model_len = None  # 让vLLM自动推断
        
        llm = AsyncLLM.from_engine_args(engine_args)
        print("✓ AsyncLLM初始化成功")
        
        # 2. 加载trace
        print("\n2. 加载trace...")
        trace_file = "/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_trace_5min_k10.jsonl"
        loader = TraceLoader(trace_file, time_range_minutes=(0, 5))
        trace_requests = loader.load_requests()
        print(f"✓ 加载了 {len(trace_requests)} 个请求")
        
        # 3. 创建HashIDMapper
        print("\n3. 创建HashIDMapper...")
        vocab_size = llm.model_config.get_vocab_size()
        mapper = HashIDMapper(vocab_size=vocab_size, seed=42)
        print(f"✓ HashIDMapper创建成功 (vocab_size={vocab_size})")
        
        # 4. 创建TraceScheduler
        print("\n4. 创建TraceScheduler...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
            ignore_eos=True,
        )
        scheduler = TraceScheduler(llm, mapper, sampling_params)
        print("✓ TraceScheduler创建成功")
        
        # 5. 运行少量请求测试
        print("\n5. 运行前3个请求...")
        test_requests = trace_requests[:20]
        
        outputs = await scheduler.schedule_requests(
            test_requests,
            realtime=True,  # 非实时模式，快速测试
        )
        
        print(f"\n✓ 完成！")
        print(f"  总请求数: {len(test_requests)}")
        print(f"  返回输出数: {len(outputs)}")
        
        # 6. 检查输出
        print("\n6. 检查输出...")
        valid_outputs = [out for out in outputs if out is not None]
        print(f"  有效输出: {len(valid_outputs)}/{len(outputs)}")
        
        if len(valid_outputs) > 0:
            print("\n✅ 成功！有有效输出")
            for i, out in enumerate(valid_outputs[:20]):
                print(f"\n  请求 {i}:")
                print(f"    finished: {out.finished}")
                print(f"    outputs数量: {len(out.outputs)}")
                if len(out.outputs) > 0:
                    print(f"    token_ids数量: {len(out.outputs[0].token_ids)}")
                    print(f"    文本片段: {out.outputs[0].text[:50]}...")
            return True
        else:
            print("\n❌ 失败！没有有效输出")
            return False
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        if 'llm' in locals():
            print("\n7. 关闭AsyncLLM...")
            llm.shutdown()
            print("✓ AsyncLLM已关闭")

if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    print("\n" + "=" * 80)
    if success:
        print("✅ 完整pipeline测试通过！")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ 完整pipeline测试失败！")
        print("=" * 80)
        sys.exit(1)

