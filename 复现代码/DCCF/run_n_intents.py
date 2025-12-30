import os

datasets = ['gowalla', 'amazon', 'tmall']
n_intents_list = [32, 64, 128, 256]

for dataset in datasets:
    for n_intents in n_intents_list:
        output = f"{dataset}_{n_intents}_"
        cmd = (
            f"python DCCF_PyTorch.py "
            f"--dataset {dataset} "
            f"--n_intents {n_intents} "
            f"--output {output}"
        )
        print("Running:", cmd)
        os.system(cmd)
