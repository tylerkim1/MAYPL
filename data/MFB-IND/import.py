entity2id = {}
relation2id = {}
id2ent_dict = {}
id2rel_dict = {}
id2ent = []
id2rel = []
train = []
msg = []
valid = []
test = []
with open("../../data_orig/MFB-IND/entities.txt") as f:
    for line in f.readlines():
        idx, entity = line.strip().split("\t")
        entity2id[entity] = idx
        id2ent_dict[idx] = entity

with open("../../data_orig/MFB-IND/relations.txt") as f:
    for line in f.readlines():
        idx, relation = line.strip().split("\t")
        relation2id[relation] = idx
        id2rel_dict[idx] = relation

ent_set = set()
rel_set = set()

with open("../../data_orig/MFB-IND/train.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        rel_split = id2rel_dict[elements[0]].split("_")
        assert len(rel_split) == len(elements)-1, (line, len(rel_split) ,  len(elements)) 
        rel_set.add("_".join(rel_split[:2]))
        for rel in rel_split[2:]:
            rel_set.add(rel)
        ents = elements[1:]
        for ent in ents:
            ent_set.add(ent)
        fact = [ents[0], "_".join(rel_split[:2]), ents[1]]
        for i, q in enumerate(rel_split[2:]):
            fact.append(q)
            fact.append(ents[2+i])
        train.append("\t".join(fact)+"\n")
        msg.append("\t".join(fact)+"\n")
with open("../../data_orig/MFB-IND/au_.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        rel_split = id2rel_dict[elements[0]].split("_")
        assert len(rel_split) == len(elements)-1, (line, len(rel_split) ,  len(elements)) 
        rel_set.add("_".join(rel_split[:2]))
        for rel in rel_split[2:]:
            rel_set.add(rel)
        ents = elements[1:]
        for ent in ents:
            ent_set.add(ent)
        fact = [ents[0], "_".join(rel_split[:2]), ents[1]]
        for i, q in enumerate(rel_split[2:]):
            fact.append(q)
            fact.append(ents[2+i])
        msg.append("\t".join(fact)+"\n")
with open("../../data_orig/MFB-IND/valid.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        rel_split = id2rel_dict[elements[0]].split("_")
        rel_set.add("_".join(rel_split[:2]))
        assert len(rel_split) == len(elements)-1, (line, len(rel_split) ,  len(elements)) 
        for rel in rel_split[2:]:
            rel_set.add(rel)
        ents = elements[1:]
        for ent in ents:
            ent_set.add(ent)
        fact = [ents[0], "_".join(rel_split[:2]), ents[1]]
        for i, q in enumerate(rel_split[2:]):
            fact.append(q)
            fact.append(ents[2+i])
        valid.append("\t".join(fact)+"\n")
with open("../../data_orig/MFB-IND/test.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        rel_split = id2rel_dict[elements[0]].split("_")
        rel_set.add("_".join(rel_split[:2]))
        assert len(rel_split) == len(elements)-1, (line, len(rel_split) ,  len(elements)) 
        for rel in rel_split[2:]:
            rel_set.add(rel)
        ents = elements[1:]
        for ent in ents:
            ent_set.add(ent)
        fact = [ents[0], "_".join(rel_split[:2]), ents[1]]
        for i, q in enumerate(rel_split[2:]):
            fact.append(q)
            fact.append(ents[2+i])
        test.append("\t".join(fact)+"\n")
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
ents = sorted(list(ent_set))
rels = sorted(list(rel_set))
with open("entities.txt", "w") as f:
    for ent in ents:
        f.write(ent + "\n")
with open("key2ent.txt", "w") as f:
    for idx in id2ent_dict:
        f.write(f"{idx}\t{id2ent_dict[idx]}\n")
with open("relations.txt", "w") as f:
    for rel in rels:
        f.write(rel + "\n")
        