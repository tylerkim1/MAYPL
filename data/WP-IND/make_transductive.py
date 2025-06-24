seen_ents = set()
with open("train.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        for ent in elements[::2]:
            seen_ents.add(ent)
with open("msg.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        for ent in elements[::2]:
            seen_ents.add(ent)
unseen_ents = set()
with open("valid.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        for ent in elements[::2]:
            if ent not in seen_ents:
                unseen_ents.add(ent)
with open("test.txt") as f:
    for line in f.readlines():
        elements = line.strip().split("\t")
        for ent in elements[::2]:
            if ent not in seen_ents:
                unseen_ents.add(ent)

with open("relations.txt", "a") as f:
    f.write(f"__SELF__\n")

with open("msg.txt", "a") as f:
    for ent in sorted(list(unseen_ents)):
        f.write(f"{ent}\t__SELF__\t{ent}\n")