def parse_input(input_text):
    rules_section, updates_section = input_text.strip().split('\n\n')
    rules = []
    for line in rules_section.strip().split('\n'):
        a, b = map(int, line.strip().split('|'))
        rules.append((a, b))
    updates = []
    for line in updates_section.strip().split('\n'):
        updates.append(list(map(int, line.strip().split(','))))
    return rules, updates

def is_valid_order(update, rules):
    index_map = {val: idx for idx, val in enumerate(update)}
    for a, b in rules:
        if a in index_map and b in index_map:
            if index_map[a] >= index_map[b]:
                return False
    return True

def find_middle(update):
    return update[len(update) // 2]

def solve(input_text):
    rules, updates = parse_input(input_text)
    total = 0
    for update in updates:
        if is_valid_order(update, rules):
            total += find_middle(update)
    return total

# Example usage
input_data = """47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47"""

print(solve(input_data))

