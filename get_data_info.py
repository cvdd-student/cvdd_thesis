from organise_data import get_data


def find_and_print_info(data, name):
    lines_amt = 0
    total_chars = 0
    amt_equal = 0
    amt_comments = 0
    amt_imports = 0
    amt_prints = 0
    amt_whitespaces = 0
    amt_function_definitions = 0
    amt_def_solve = 0
    for item, label in data:
        for line in item.split("\n"):
            lines_amt += 1
            total_chars += len(line)
            if "def " in line:
                amt_function_definitions += 1
            if "import " in line:
                amt_imports += 1
            if "print(" in line:
                amt_prints += 1
            for char in line:
                if char == "=":
                    amt_equal += 1
                elif char == "#":
                    amt_comments += 1
                elif char == " ":
                    amt_whitespaces += 1
        
        if "def solve()" in item:
            amt_def_solve += 1
    
    average_chars_per_line = total_chars / lines_amt
    average_lines_per_file = lines_amt / len(data)
    average_equal_per_line = amt_equal / lines_amt
    average_defs_per_file = amt_function_definitions / len(data)
    average_comments_per_file = amt_comments / len(data)
    average_imports_per_file = amt_imports / len(data)
    average_prints_per_file = amt_prints / len(data)
    average_whitespaces_per_line = amt_whitespaces / lines_amt
    
    print(name.upper() + " DATA INFORMATION")
    print("Avg. chars per line: " + str(average_chars_per_line))
    print("Avg. lines per file: " + str(average_lines_per_file))
    print("Avg. equal signs per file: " + str(average_equal_per_line))
    print("Avg. function definitions per file: " + str(average_defs_per_file))
    print("Avg. comments per file: " + str(average_comments_per_file))
    print("Avg. imports per file: " + str(average_imports_per_file))
    print("Avg. prints per file: " + str(average_prints_per_file))
    print("Avg. whitespaces per line: " + str(average_whitespaces_per_line))
    print("def solve(): " + str(amt_def_solve))
    print()


def main():
    gemini_data = get_data("Gemini_Data", "gemini")
    human_data = get_data("Human_Data", "human")
    
    find_and_print_info(gemini_data, "gemini")
    find_and_print_info(human_data, "human")


if __name__ == "__main__":
    main()