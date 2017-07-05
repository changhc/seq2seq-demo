import random
import sys 

start = 65  # 'A'
sample_count = 8 

def gen_random():
    letters = random.sample(range(start, start + 26), sample_count)
        return ''.join([chr(x) for x in letters])

def main():
    seq_count = int(1e5)
    seq_list = []
    for i in range(seq_count):
        if sys.argv[1] != '3':
             seq_list.append(gen_random())
    with open('train-' + sys.argv[1], 'w') as file:
        for seq in seq_list:
            file.write(seq + "\n")
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Invalid arguments!")
    main()

