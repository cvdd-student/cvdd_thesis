import random
import time

def simulate_horse_race(num_horses=5):
    """Simulates a horse race.

    Args:
        num_horses: The number of horses in the race.

    Returns:
        A dictionary containing the finishing positions of each horse and the race time.
    """

    horses = {}
    for i in range(num_horses):
        horses[f"Horse {i+1}"] = 0  # Initialize position to 0 (start)

    race_time = 0  # Initialize race time
    while True:
        race_time += random.uniform(0.1, 0.5)  # Simulate variable run speeds
        
        # Avoid infinite loop if a horse is stuck; small chance of tie
        positions_changed = False
        for horse, pos in horses.items():
            move_distance = random.uniform(0, 1)  # Simulate speed
            horses[horse] = pos + move_distance
            positions_changed = True
        
        # Check if a horse has finished (passed 10 or the threshold is reached)
        if any(pos >= 10 for pos in horses.values()):
            break

    # Convert positions to ranks, handle potential ties
    finished_positions = {}
    sorted_horses = sorted(horses.items(), key=lambda x: x[1])
    rank = 1
    last_pos = -1
    for horse, pos in sorted_horses:
        if pos > last_pos:
            rank = rank
            last_pos = pos
        else:
          rank = rank + 1 - sorted_horses[sorted_horses.index((horse, pos)) - 1][1]
        finished_positions[horse] = rank



    return finished_positions, race_time


def display_results(results, race_time):
    """Displays the race results."""

    print(f"\nRace Results (Time: {race_time:.2f} seconds):")
    for horse, position in results.items():
        print(f"{horse:<15} - {position}th place")
    print("----------------------")


# Example usage:
num_horses = 5
results, race_time = simulate_horse_race(num_horses)
display_results(results, race_time)
