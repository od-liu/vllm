#!/usr/bin/env python3
"""
测试hash_id映射是否产生有效的token_ids
"""

import json
import sys

# 添加vllm到路径
sys.path.insert(0, '/mnt/disk1/ljm/vllm')

from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceLoader

def test_hash_mapping():
    print("=" * 80)
    print("测试hash_id到token_id的映射")
    print("=" * 80)
    
    # 1. 加载trace文件
    print("\n1. 加载trace文件...")
    trace_file = "/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_trace_5min_k10.jsonl"
    
    loader = TraceLoader(trace_file, time_range_minutes=(0, 5))
    requests = loader.load_requests()
    
    print(f"✓ 加载了 {len(requests)} 个请求")
    
    # 2. 创建HashIDMapper
    print("\n2. 创建HashIDMapper...")
    vocab_size = 151936  # Qwen模型的vocab_size
    mapper = HashIDMapper(vocab_size=vocab_size, seed=42)
    print(f"✓ HashIDMapper创建成功 (vocab_size={vocab_size})")
    
    # 3. 测试前几个请求的映射
    print("\n3. 测试hash_id映射...")
    print("-" * 80)
    
    for i in range(min(5, len(requests))):
        req = requests[i]
        print(f"\n请求 {i}:")
        print(f"  chat_id: {req.chat_id}")
        print(f"  hash_ids数量: {len(req.hash_ids)}")
        print(f"  hash_ids前5个: {req.hash_ids[:5]}")
        print(f"  input_length: {req.input_length}")
        print(f"  output_length: {req.output_length}")
        
        # 映射hash_ids到token_ids
        token_ids = mapper.map_hash_ids(req.hash_ids)
        
        print(f"\n  映射结果:")
        print(f"    token_ids类型: {type(token_ids)}")
        print(f"    token_ids数量: {len(token_ids)}")
        
        if len(token_ids) > 0:
            print(f"    token_ids前5个: {token_ids[:5]}")
            print(f"    token_ids类型检查: {type(token_ids[0])}")
            
            # 检查是否都是整数
            all_ints = all(isinstance(t, int) for t in token_ids[:10])
            print(f"    都是整数? {all_ints}")
            
            # 检查范围
            if all_ints:
                min_token = min(token_ids)
                max_token = max(token_ids)
                print(f"    token范围: {min_token} - {max_token}")
                
                # 检查是否在vocab范围内
                if max_token >= vocab_size:
                    print(f"    ⚠️  警告: 有token超出vocab_size范围！")
                    print(f"       max_token={max_token}, vocab_size={vocab_size}")
                else:
                    print(f"    ✓ 所有token都在vocab范围内")
        else:
            print(f"    ❌ token_ids为空！")
        
        # 截断到input_length
        if req.input_length > 0:
            truncated_tokens = token_ids[:req.input_length]
            print(f"\n  截断到input_length={req.input_length}:")
            print(f"    截断后数量: {len(truncated_tokens)}")
            
            if len(truncated_tokens) == 0:
                print(f"    ❌ 截断后为空！这会导致生成失败！")
            else:
                print(f"    ✓ 截断后有效")
    
    # 4. 统计潜在问题
    print("\n" + "=" * 80)
    print("4. 统计分析")
    print("=" * 80)
    
    empty_count = 0
    invalid_count = 0
    
    for req in requests[:50]:  # 检查前50个
        token_ids = mapper.map_hash_ids(req.hash_ids)
        
        if req.input_length > 0:
            truncated = token_ids[:req.input_length]
            if len(truncated) == 0:
                empty_count += 1
        
        if not all(isinstance(t, int) for t in token_ids):
            invalid_count += 1
    
    print(f"\n检查了前50个请求:")
    print(f"  截断后为空的请求: {empty_count}")
    print(f"  包含非整数token的请求: {invalid_count}")
    
    if empty_count > 0 or invalid_count > 0:
        print(f"\n❌ 发现问题！这些请求会导致生成失败")
        return False
    else:
        print(f"\n✅ 所有检查通过")
        return True

if __name__ == "__main__":
    try:
        success = test_hash_mapping()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)






