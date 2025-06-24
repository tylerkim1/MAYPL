ents = set()
rels = set()
train = []
valid = []
test = []
with open("../../data_orig/wd50k/statements/train.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        train.append(line.replace(",","\t"))
with open("../../data_orig/wd50k/statements/valid.txt") as f:
    for line in f.readlines():
        elements = line.strip().split(",")
        for ent in elements[::2]:
            ents.add(ent)
        for rel in elements[1::2]:
            rels.add(rel)
        valid.append(line.replace(",","\t"))
with open("../../data_orig/wd50k/statements/test.txt") as f:
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
        