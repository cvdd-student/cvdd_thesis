def count_characters(filepath):
    """Counts the occurrences of each character in a text file.

    Args:
        filepath: The path to the text file.

    Returns:
        A dictionary where keys are characters and values are their counts.
        Returns None if the file doesn't exist or is not readable.
    """

    try:
        with open(filepath, 'r') as file:
            char_counts = {}
            for line in file:
                for char in line:
                    char_counts[char] = char_counts.get(char, 0) + 1
            return char_counts
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    