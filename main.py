import subprocess

for label_ratio in [0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f"========================= Running pre-fine.py with label_ratio={label_ratio} =============================")
    subprocess.run(["python", "pre-fine.py", '--label-ratio', f'{label_ratio}'])
    print(f"========================= Finished running pre-fine.py with label_ratio={label_ratio}=====================")