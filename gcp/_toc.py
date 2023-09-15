# parse the document for titles
titles = []
with open('README.md') as file:
    for line in file:
        if line.startswith("###"):
            titles.append(("subtitle", line[4:-1]))
        elif line.startswith("##"):
            titles.append(("title", line[3:-1]))

# print the toc
for type_, title in titles:
    link = ''.join(
        c for c in title.lower().replace(" ","-") if c.isalnum() or c=="-"
    )
    prefix = "" if type_ == "title" else "  "
    print(f"{prefix}* [{title}](#{link})")

