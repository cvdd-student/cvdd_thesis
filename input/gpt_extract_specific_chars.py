from collections import Counter
import string

def count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    counter = Counter(text)
    question_marks = counter['?']
    equal_signs = counter['=']
    lowercase_letters = sum(counter[c] for c in string.ascii_lowercase)
    return question_marks + equal_signs + lowercase_letters

# Example usage:
# total = count_characters('input.txt')
# print(total)
