import random
import sys 

start = 65  # 'A'
sample_count = 8 
alphabet = range(start, start + 26)
vowel = [start + 0, start + 4, start + 8, start + 14, start + 20]
consonant = [x for x in alphabet if x not in vowel]

def gen_random():
    letters = random.sample(alphabet, sample_count)
    return ''.join([chr(x) for x in letters])

def gen_vowel():
    letters = random.sample(consonant, sample_count - 1)
    string = ''.join([chr(x) for x in letters])
    vowel_position = random.randint(0, sample_count - 1)
    choice = random.choice(vowel)
    return [string[:vowel_position] + chr(choice) + string[vowel_position:], choice]

def main():
    seq_count = int(1e5)
    seq_list = []
    for i in range(seq_count):
        if sys.argv[1] != '3':
            seq_list.append(gen_random())
        else:
            seq_list.append(gen_vowel())
    with open('train-x-' + sys.argv[1], 'w') as file:
        if sys.argv[1] != '3':
            for seq in seq_list:
                file.write(seq + "\n")
        else:
            for seq in seq_list:
                file.write(seq[0] + "\n")
    with open('train-y-' + sys.argv[1], 'w') as file:
        if sys.argv[1] == '1':
            for seq in seq_list:
                file.write(seq + "\n")
        elif sys.argv[1] == '2':
            for seq in seq_list:
                file.write(seq[::-1] + "\n")
        elif sys.argv[1] == '3':
            for seq in seq_list:
                out = random.sample(alphabet, sample_count)
                is_vowel = [i for i in range(sample_count) if out[i] in vowel][::-1]
                for i in is_vowel:
                    out = out[:i + 1] + [seq[1]] + out [i + 1:]
                out = out[:sample_count]
                file.write(''.join([chr(x) for x in out]) + "\n")
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Invalid arguments!")
    main()

