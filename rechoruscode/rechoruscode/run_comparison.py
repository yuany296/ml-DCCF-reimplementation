
import os
import subprocess
import time


project_root = "/mnt/workspace/rechoruscode"
os.chdir(project_root)


models = ["BPRMF", "LightGCN", "DCCF"]
datasets = ["Grocery_and_Gourmet_Food", "MovieLens_1M"]
seed = 0


model_params = {
    "BPRMF": "--emb_size 128 --lr 0.001 --l2 1e-4 --num_neg 1",
    "LightGCN": "--emb_size 64 --n_layers 4 --lr 0.0005 --l2 1e-4 --num_neg 1",
    "DCCF": "--emb_size 64 --n_layers 3 --n_intents 4 --temp 0.2 --emb_reg 1e-3 --cen_reg 1e-3 --ssl_reg 0.05 --l2 0 --num_neg 1"
}


common_params = "--epoch 100 --early_stop 20 --test_epoch 5 --batch_size 512 --eval_batch_size 512 --topk '5,10,20,50' --main_metric 'NDCG@10' --train 1 --load 0"

print("="*80)
print(f"项目根目录: {project_root}")
print(f"当前目录: {os.getcwd()}")
print("="*80)

for dataset in datasets:
    print(f"\n数据集: {dataset}")
    print("-"*80)
    
    for model in models:
        print(f"\n模型: {model}")
        print("-"*60)
        
        cmd = f"python src/main.py --model_name {model} --dataset {dataset} --random_seed {seed} {model_params[model]} {common_params}"
        
        start_time = time.time()
        try:
            print(f"运行命令: {cmd}")
            
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            
            for line in process.stdout:
                line = line.strip()
                if any(keyword in line for keyword in [
                    "Epoch", "Test", "Dev", "Best Iter", "Save model", "Load model",
                    "BEGIN", "END", "Device:", "Reading data", "#params:", "Optimizer:"
                ]):
                    print(f"  {line}")
            
            process.wait()
            
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\n✓ {model} 完成，耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
            else:
                print(f"\n✗ {model} 失败，退出码: {process.returncode}")
                
        except Exception as e:
            print(f"\n✗ {model} 异常: {e}")

print("\n" + "="*80)
print("所有实验完成！")
print("="*80)