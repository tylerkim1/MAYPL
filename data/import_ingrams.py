import os

for dataset_name in ['WK', 'NL', 'FB']:
    for version in ['100', '75', '50', '25']:
        os.makedirs(f"./{dataset_name}-{version}", exist_ok = True)
        train = []
        msg = []
        valid = []
        test = []
        with open(f"../data_orig/{dataset_name}-{version}/train.txt") as f:
            for line in f.readlines():
                train.append(line.replace("_", "-"))
        with open(f"../data_orig/{dataset_name}-{version}/msg.txt") as f:
            for line in f.readlines():
                msg.append(line.replace("_", "-"))
        with open(f"../data_orig/{dataset_name}-{version}/valid.txt") as f:
            for line in f.readlines():
                valid.append(line.replace("_", "-"))
        with open(f"../data_orig/{dataset_name}-{version}/test.txt") as f:
            for line in f.readlines():
                test.append(line.replace("_", "-"))
        with open(f"./{dataset_name}-{version}/train.txt", "w") as f:
            for line in train:
                f.write(line)
        with open(f"./{dataset_name}-{version}/msg.txt", "w") as f:
            for line in msg:
                f.write(line)
        with open(f"./{dataset_name}-{version}/valid.txt", "w") as f:
            for line in valid:
                f.write(line)
        with open(f"./{dataset_name}-{version}/test.txt", "w") as f:
            for line in test:
                f.write(line)
