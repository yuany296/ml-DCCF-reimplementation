# simple_ablation_runner.py
import os
import subprocess
import time

def run_single_experiment(dataset, ablation_type, epochs=100):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"开始运行: {dataset} - {ablation_type}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'DCCF_PyTorch.py',
        '--dataset', dataset,
        '--ablation', ablation_type,
        '--output', f"{dataset}_{ablation_type}",
        '--epoch', str(epochs)
    ]
    
    # 执行命令
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        print(f"运行完成，耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"运行出错: {e}")
        return False

def run_all_experiments():
    """运行所有消融实验"""
    
    # 配置
    datasets = ['gowalla','amazon']
    ablation_types = [
        'wo_disen', 
        'wo_local_r', 
        'wo_disen_r',   
        'wo_ssl_disen',  
        'wo_all_ada', 
        'full',                 
    ]
    
    # 创建必要的文件夹
    os.makedirs('log', exist_ok=True)
    os.makedirs('saved', exist_ok=True)
    
    # 运行所有实验
    results = {}
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"处理数据集: {dataset}")
        print(f"{'#'*70}")
        
        for ablation_type in ablation_types:
            success = run_single_experiment(dataset, ablation_type, epochs=100)  # 先用50轮测试
            results[f"{dataset}_{ablation_type}"] = success
            
            # 避免连续运行导致过热，可以添加短暂延迟
            time.sleep(2)
    
    # 汇总结果
    print("\n" + "="*70)
    print("实验运行完成汇总:")
    print("="*70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("失败的实验:")
        for exp, success in results.items():
            if not success:
                print(f"  - {exp}")
    
    return results

if __name__ == '__main__':
    print("开始DCCF消融实验...")
    results = run_all_experiments()