import torch
import os
import argparse

def load_state_dict(checkpoint_path):
    """加载checkpoint并提取state_dict"""
    print(f"正在加载: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取state_dict
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        # 如果整个checkpoint就是state_dict
        state_dict = ckpt
    
    return state_dict

def get_top_level_prefixes(state_dict):
    """获取state_dict的第一层前缀"""
    prefixes = set()
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        prefixes.add(prefix)
    return sorted(list(prefixes))

def merge_models(finetuned_path, original_path, output_path):
    """
    合并微调后的模型和原始模型的tracker部分
    
    Args:
        finetuned_path: 微调后的模型路径
        original_path: 原始sam3模型路径
        output_path: 输出模型路径
    """
    # 1. 加载微调后的模型
    print("\n=== 步骤1: 加载微调后的模型 ===")
    finetuned_state_dict = load_state_dict(finetuned_path)
    finetuned_prefixes = get_top_level_prefixes(finetuned_state_dict)
    print(f"微调后模型的第一层前缀: {finetuned_prefixes}")
    
    # 2. 加载原始模型
    print("\n=== 步骤2: 加载原始sam3模型 ===")
    original_state_dict = load_state_dict(original_path)
    original_prefixes = get_top_level_prefixes(original_state_dict)
    print(f"原始模型的第一层前缀: {original_prefixes}")
    
    # 3. 构建新的state_dict
    print("\n=== 步骤3: 构建合并后的模型 ===")
    merged_state_dict = {}
    
    # 3.1 将微调后的模型的所有键加上"detector."前缀
    print("正在为微调后的模型添加'detector.'前缀...")
    for key, value in finetuned_state_dict.items():
        new_key = f"detector.{key}"
        merged_state_dict[new_key] = value
        if len(merged_state_dict) <= 5:  # 打印前5个键作为示例
            print(f"  {key} -> {new_key}")
    
    print(f"已添加 {len(finetuned_state_dict)} 个参数（带detector前缀）")
    
    # 3.2 从原始模型中提取tracker部分
    print("\n正在从原始模型中提取tracker部分...")
    tracker_keys = [k for k in original_state_dict.keys() if k.startswith('tracker.')]
    print(f"找到 {len(tracker_keys)} 个tracker相关的参数")
    
    if len(tracker_keys) == 0:
        print("警告: 原始模型中未找到tracker相关的参数！")
    else:
        for key in tracker_keys:
            merged_state_dict[key] = original_state_dict[key]
            if len([k for k in merged_state_dict.keys() if k.startswith('tracker.')]) <= 5:
                print(f"  {key}")
    
    # 4. 验证合并结果
    print("\n=== 步骤4: 验证合并结果 ===")
    merged_prefixes = get_top_level_prefixes(merged_state_dict)
    print(f"合并后模型的第一层前缀: {merged_prefixes}")
    
    # 统计各模块的参数数量
    print("\n各模块参数统计:")
    for prefix in merged_prefixes:
        count = sum(1 for k in merged_state_dict.keys() if k.startswith(prefix + '.'))
        param_count = sum(v.numel() for k, v in merged_state_dict.items() if k.startswith(prefix + '.'))
        print(f"  - {prefix}: {count} 个参数, 参数量: {param_count}")
    
    # 5. 保存合并后的模型
    print(f"\n=== 步骤5: 保存合并后的模型 ===")
    print(f"保存到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存state_dict
    torch.save(merged_state_dict, output_path)
    
    print("✓ 合并完成！")
    print(f"总参数数量: {len(merged_state_dict)}")
    print(f"总参数量: {sum(v.numel() for v in merged_state_dict.values())}")

def main():
    parser = argparse.ArgumentParser(description='合并微调后的模型和原始模型的tracker部分')
    parser.add_argument(
        '--finetuned',
        type=str,
        default='/home/gaoqi/sam3/gq_experiment/acdc_camus/full/1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP_model_only.pt',
        help='微调后的模型路径'
    )
    parser.add_argument(
        '--original',
        type=str,
        default='/home/gaoqi/official_ckpt/sam3_hf/sam3.pt',
        help='原始sam3模型路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出模型路径（默认为微调模型同目录下的merged_model.pt）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，使用默认路径
    if args.output is None:
        output_dir = os.path.dirname(args.finetuned)
        args.output = os.path.join(output_dir, 'merged_model_with_tracker.pt')
    
    merge_models(args.finetuned, args.original, args.output)

if __name__ == '__main__':
    main()
