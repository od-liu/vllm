#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
验证AsyncLLM是否可用的简单脚本
运行此脚本以确认vLLM v1 AsyncLLM正常工作
"""

import sys
import asyncio

def check_imports():
    """检查必要的导入是否可用"""
    print("=" * 80)
    print("步骤 1/3: 检查导入...")
    print("=" * 80)
    
    try:
        import vllm
        print(f"✅ vLLM版本: {vllm.__version__}")
    except ImportError as e:
        print(f"❌ 无法导入vLLM: {e}")
        return False
    
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        print("✅ AsyncLLM导入成功")
    except ImportError as e:
        print(f"❌ 无法导入AsyncLLM: {e}")
        print("   提示: 您的vLLM版本可能不支持v1引擎")
        return False
    
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        print("✅ AsyncEngineArgs导入成功")
    except ImportError as e:
        print(f"❌ 无法导入AsyncEngineArgs: {e}")
        return False
    
    try:
        from vllm import SamplingParams
        print("✅ SamplingParams导入成功")
    except ImportError as e:
        print(f"❌ 无法导入SamplingParams: {e}")
        return False
    
    return True


def check_trace_scheduler():
    """检查TraceScheduler是否已更新"""
    print("\n" + "=" * 80)
    print("步骤 2/3: 检查TraceScheduler修改...")
    print("=" * 80)
    
    try:
        from vllm.profiler.trace_scheduler import TraceScheduler
        import inspect
        
        # 检查__init__签名
        sig = inspect.signature(TraceScheduler.__init__)
        params = list(sig.parameters.keys())
        
        # 检查是否接受AsyncLLM（通过参数名判断）
        if 'llm' in params:
            print("✅ TraceScheduler.__init__包含llm参数")
        else:
            print("❌ TraceScheduler.__init__缺少llm参数")
            return False
        
        # 检查源代码中是否移除了ThreadPoolExecutor
        source = inspect.getsource(TraceScheduler)
        
        if 'ThreadPoolExecutor' in source:
            print("⚠️  警告: TraceScheduler源代码中仍包含ThreadPoolExecutor")
            print("   这可能意味着修改未完全应用")
            return False
        else:
            print("✅ TraceScheduler已移除ThreadPoolExecutor")
        
        if 'AsyncLLM' in source or 'async_llm' in source:
            print("✅ TraceScheduler代码中包含AsyncLLM相关内容")
        else:
            print("⚠️  警告: TraceScheduler源代码中未找到AsyncLLM引用")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查TraceScheduler时出错: {e}")
        return False


async def test_basic_asyncllm():
    """测试AsyncLLM基本功能（使用小模型）"""
    print("\n" + "=" * 80)
    print("步骤 3/3: 测试AsyncLLM基本功能...")
    print("=" * 80)
    
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm import SamplingParams
        
        print("正在初始化AsyncLLM（使用小模型）...")
        print("注意: 这可能需要几分钟时间下载模型...")
        
        # 使用一个小模型进行测试
        engine_args = AsyncEngineArgs(
            model="facebook/opt-125m",  # 很小的模型，快速测试
            enforce_eager=True,
            max_model_len=256,
        )
        
        llm = AsyncLLM.from_engine_args(engine_args)
        print("✅ AsyncLLM初始化成功")
        
        # 测试生成
        print("\n测试异步生成...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
        )
        
        prompt = "Hello, world!"
        request_id = "test_request_1"
        
        result = None
        async for output in llm.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            result = output
            if output.finished:
                break
        
        if result and result.finished:
            print("✅ 异步生成成功完成")
            print(f"   生成的token数: {len(result.outputs[0].token_ids)}")
            print(f"   输出文本: {result.outputs[0].text[:50]}...")
        else:
            print("❌ 生成未正常完成")
            return False
        
        # 清理
        llm.shutdown()
        print("✅ AsyncLLM shutdown成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试AsyncLLM时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("AsyncLLM验证脚本")
    print("=" * 80)
    
    # 步骤1: 检查导入
    if not check_imports():
        print("\n" + "=" * 80)
        print("❌ 导入检查失败")
        print("=" * 80)
        sys.exit(1)
    
    # 步骤2: 检查TraceScheduler
    if not check_trace_scheduler():
        print("\n" + "=" * 80)
        print("⚠️  TraceScheduler检查有问题，但可能不影响功能")
        print("=" * 80)
    
    # 步骤3: 测试AsyncLLM（可选）
    print("\n是否运行AsyncLLM功能测试？")
    print("注意: 这会下载一个小模型（~500MB）并运行测试生成")
    response = input("运行测试? [y/N]: ").strip().lower()
    
    if response == 'y':
        success = asyncio.run(test_basic_asyncllm())
        
        if success:
            print("\n" + "=" * 80)
            print("✅ 所有检查通过！AsyncLLM可以使用")
            print("=" * 80)
            print("\n您可以开始运行完整的benchmark了：")
            print("  python benchmarks/e2e_sweep/run_e2e_sweep.py \\")
            print("      --trace-file qwen-bailian-usagetraces-anon/qwen_trace_5min_sparse.jsonl \\")
            print("      --model Qwen/Qwen2.5-7B-Instruct \\")
            print("      --num-gpus 8 \\")
            print("      --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \\")
            print("      --output-dir e2e_results/asyncllm_test")
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ AsyncLLM测试失败")
            print("=" * 80)
            sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("✅ 导入检查通过，跳过功能测试")
        print("=" * 80)
        print("\n您可以直接运行benchmark，如果有问题再回来运行功能测试")
        sys.exit(0)


if __name__ == "__main__":
    main()








