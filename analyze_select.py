import os
import sys

dataset = sys.argv[1]
filename = f"logs/select_{sys.argv[3]}/{dataset}/our_50_llama2-{sys.argv[2]}b.log"
f = open(filename, "r")
pred_all = []
index_list = []
while True:
    line = f.readline()
    if not line:
        break
    if "uncertain_indices" in line:
        length = int(line[line.index("(") + 1 : line.index(")")])
        if "]" in line:
            line = line[line.index("[") + 1 : line.index("]")]
        else:
            line = line[line.index("[") + 1 :]
        line = line.strip()
        uncertain_indices = line.replace("  ", " ").split(" ")
        while len(uncertain_indices) < length:
            line = f.readline()
            if "]" in line:
                line = line[: line.index("]")]
            line = line.strip()
            uncertain_indices += line.replace("  ", " ").split(" ")
    if "index: " in line:
        index = line.split("index: ")[1].split(",")[0]
        index_list.append(index)
        line = f.readline()
        if "pred_last" in line:
            line = f.readline()
        result = eval(line)
        _, labels, probs = zip(*result)
        labels = list(labels)
        probs = list(probs)
        labels_output, probs_output = [], []
        for i, idx in enumerate(uncertain_indices):
            idx = int(idx)
            labels_output.append(str(labels[idx]))
            probs_output.append(str(probs[idx]))
        print(" ".join(uncertain_indices))
        print(" ".join(labels_output))
        print(" ".join(probs_output))
        print(f"index={index},label={labels[int(index)]},prob={probs[int(index)]}")
        print()
f.close()
