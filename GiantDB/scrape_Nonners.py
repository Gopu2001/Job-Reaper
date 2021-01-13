# Anmol Kapoor

with open('non_jobs.txt', mode='r', encoding='utf8') as file:
    content = file.readlines()

non_jobs = ""
for cont in content:
    if cont.strip() != '' and cont not in non_jobs:
        non_jobs += cont.strip() + "\n"
#x = ''
#for i in non_jobs: x += i + '\n'

file = open("output_non_jobs.txt", mode="w", encoding="utf-8")
file.write(non_jobs)
file.close()
