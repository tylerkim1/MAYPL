import json
ents = set()
rels = set()
train = []
valid = []
test = []
with open("../../data_orig/WikiPeople/n-ary_train.json") as f:
    fact_dicts = f.readlines()
    for fact_dict in fact_dicts:
        fact_dict = json.loads(fact_dict)
        fact_seq_dict = {"qual":[]}
        fact = []
        for r in fact_dict:
            if r == "N":
                continue
            if "_h" in r:
                fact_seq_dict["h"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            elif "_t" in r:
                fact_seq_dict["t"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            else:
                for val in fact_dict[r]:
                    if val[0] != "Q":
                        continue
                    fact_seq_dict["qual"].append((r, val))
        if fact_seq_dict["t"][0] != "Q":
            continue
        sorted_quals = sorted(fact_seq_dict["qual"])
        fact.append(fact_seq_dict["h"])
        ents.add(fact_seq_dict["h"])
        fact.append(fact_seq_dict["r"])
        rels.add(fact_seq_dict["r"])
        fact.append(fact_seq_dict["t"])
        ents.add(fact_seq_dict["t"])
        for qual in sorted_quals:
            rels.add(qual[0])
            ents.add(qual[1])
            fact.append(qual[0])
            fact.append(qual[1])
        train.append("\t".join(fact)+"\n")
with open("../../data_orig/WikiPeople/n-ary_valid.json") as f:
    fact_dicts = f.readlines()
    for fact_dict in fact_dicts:
        fact_dict = json.loads(fact_dict)
        fact_seq_dict = {"qual":[]}
        fact = []
        for r in fact_dict:
            if r == "N":
                continue
            if "_h" in r:
                fact_seq_dict["h"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            elif "_t" in r:
                fact_seq_dict["t"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            else:
                for val in fact_dict[r]:
                    if val[0] != "Q":
                        continue
                    fact_seq_dict["qual"].append((r, val))
        if fact_seq_dict["t"][0] != "Q":
            continue
        sorted_quals = sorted(fact_seq_dict["qual"])
        fact.append(fact_seq_dict["h"])
        ents.add(fact_seq_dict["h"])
        fact.append(fact_seq_dict["r"])
        rels.add(fact_seq_dict["r"])
        fact.append(fact_seq_dict["t"])
        ents.add(fact_seq_dict["t"])
        for qual in sorted_quals:
            rels.add(qual[0])
            ents.add(qual[1])
            fact.append(qual[0])
            fact.append(qual[1])
        valid.append("\t".join(fact)+"\n")
with open("../../data_orig/WikiPeople/n-ary_test.json") as f:
    fact_dicts = f.readlines()
    for fact_dict in fact_dicts:
        fact_dict = json.loads(fact_dict)
        fact_seq_dict = {"qual":[]}
        fact = []
        for r in fact_dict:
            if r == "N":
                continue
            if "_h" in r:
                fact_seq_dict["h"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            elif "_t" in r:
                fact_seq_dict["t"] = fact_dict[r]
                if "r" not in fact_seq_dict:
                    fact_seq_dict["r"] = r[:-2]
                assert fact_seq_dict["r"] == r[:-2]
            else:
                for val in fact_dict[r]:
                    if val[0] != "Q":
                        continue
                    fact_seq_dict["qual"].append((r, val))
        if fact_seq_dict["t"][0] != "Q":
            continue
        sorted_quals = sorted(fact_seq_dict["qual"])
        fact.append(fact_seq_dict["h"])
        ents.add(fact_seq_dict["h"])
        fact.append(fact_seq_dict["r"])
        rels.add(fact_seq_dict["r"])
        fact.append(fact_seq_dict["t"])
        ents.add(fact_seq_dict["t"])
        for qual in sorted_quals:
            rels.add(qual[0])
            ents.add(qual[1])
            fact.append(qual[0])
            fact.append(qual[1])
        test.append("\t".join(fact)+"\n")
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
        