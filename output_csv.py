import csv

with open('C:\\Users\\Christopher\\Desktop\\icebergTECH\\data_example\\processed\\sample_submission.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        print(','.join(row))

with open('eggs.csv', 'w', newline = '') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['SPA_M','WOW!'])


with open('eggs.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(','.join(row))

