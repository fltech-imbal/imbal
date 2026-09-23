FILE_PATH = 'results/results.txt'

NUM_VALUES_PER_ROW = 7

with open(FILE_PATH, 'r') as f:
    lines = f.readlines()

    count = 0
    totals = [0 for i in range(NUM_VALUES_PER_ROW)]
    started = False
    for line in lines:
        if ':' in line:
            if started:
                print([round(x / count, 4) for x in totals], '\n')
            print(line.strip())
            count = 0
            totals = [0 for i in range(NUM_VALUES_PER_ROW)]
            started = True
        elif '-' == line[0]:
            tokens = line.split()
            count += 1
            for i in range(NUM_VALUES_PER_ROW  + 1):
                if i == 0:
                    continue
                if tokens[i] != 'None':
                    totals[i-1] += float(tokens[i])

print([round(x / count, 4) for x in totals])