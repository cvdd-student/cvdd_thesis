def analyze_text_file(filepath):
    results = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                equals = line.count('=')
                question_marks = line.count('?')
                caps = sum(1 for char in line if char.isupper())
                results.append((equals, question_marks, caps))
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None  # Indicate failure
    return results