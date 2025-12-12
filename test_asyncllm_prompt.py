#!/usr/bin/env python3
"""
测试AsyncLLM是否能正确接受TokensPrompt输入
"""

import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.inputs import TokensPrompt

async def test_asyncllm():
    print("=" * 80)
    print("测试AsyncLLM + TokensPrompt")
    print("=" * 80)
    
    # 创建AsyncLLM
    print("\n1. 初始化AsyncLLM...")
    engine_args = AsyncEngineArgs(
        model="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )
    
    llm = AsyncLLM.from_engine_args(engine_args)
    print("✓ AsyncLLM初始化成功")
    
    # 创建TokensPrompt
    print("\n2. 创建TokensPrompt...")
    token_ids = [1, 2, 3, 4, 5]  # 简单的token序列
    prompt = TokensPrompt(prompt_token_ids=token_ids)
    print(f"✓ TokensPrompt创建成功: {prompt}")
    print(f"   类型: {type(prompt)}")
    
    # 创建SamplingParams
    print("\n3. 创建SamplingParams...")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        ignore_eos=True,
    )
    print(f"✓ SamplingParams创建成功")
    
    # 测试生成
    print("\n4. 测试异步生成...")
    request_id = "test_request_1"
    
    try:
        result = None
        async for output in llm.generate(
            request_id=request_id,
            prompt=prompt,  # 使用TokensPrompt
            sampling_params=sampling_params,
        ):
            print(f"   收到输出: finished={output.finished}")
            result = output
            if output.finished:
                break
        
        if result:
            print(f"✓ 生成成功！")
            print(f"   生成的token数: {len(result.outputs[0].token_ids)}")
            print(f"   输出文本: {result.outputs[0].text[:100]}")
        else:
            print("❌ 生成失败：没有收到输出")
            return False
            
    except Exception as e:
        print(f"❌ 生成失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 清理
    print("\n5. 关闭AsyncLLM...")
    llm.shutdown()
    print("✓ AsyncLLM已关闭")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！AsyncLLM可以正确接受TokensPrompt")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_asyncllm())
    exit(0 if success else 1)

