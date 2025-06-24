ents = set()
rels = set()
train = []
msg = []
valid = []
test = []
with open("../../data_orig/FI_WD20K100/statements/v1/transductive_train.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        train.append(line.replace(",","\t"))
with open("../../data_orig/FI_WD20K100/statements/v1/inductive_train.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        msg.append(line.replace(",","\t"))
with open("../../data_orig/FI_WD20K100/statements/v1/inductive_val.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        valid.append(line.replace(",","\t"))
with open("../../data_orig/FI_WD20K100/statements/v1/inductive_ts.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        test.append(line.replace(",","\t"))
with open("train.txt", "w") as f:
    for line in train:
        f.write(line)
with open("msg.txt", "w") as f:
    for line in msg:
        f.write(line)
with open("valid.txt", "w") as f:
    for line in valid:
        f.write(line)
with open("test.txt", "w") as f:
    for line in test:
        f.write(line)
ents = sorted(list(ents))
rels = sorted(list(rels))
with open("entities.txt", "w") as f:
    for ent in ents:
        f.write(ent + "\n")
with open("relations.txt", "w") as f:
    for rel in rels:
        f.write(rel + "\n")
        