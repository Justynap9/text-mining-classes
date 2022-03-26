import csv


def read_csv(file: str) -> str:
    file = open(file, encoding="UTF8")
    reader = csv.reader(file)
    text = '\t'.join([i[0] for i in reader])
    return text
