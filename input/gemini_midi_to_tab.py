import pretty_midi
import music21

def midi_to_tab(midi_file, output_file="tab.txt"):
    """
    Converts a MIDI file to a guitar tab representation.

    Args:
        midi_file: Path to the MIDI file.
        output_file: Path to save the tab output (default: tab.txt).
    
    Returns:
        True if successful, False otherwise.  Prints error messages on failure.
    """

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return False


    # Find the instrument most likely to be guitar.  (Crucial improvement)
    guitar_instrument = None
    for instrument in midi_data.instruments:
      if instrument.name.lower() in ('guitar', 'electric guitar', 'acoustic guitar'):
          guitar_instrument = instrument
          break
    if guitar_instrument is None:
        print("No guitar instrument found in the MIDI file.")
        return False


    # Extract notes and their timing information.  Critical for accurate timing.
    notes = []
    for note in guitar_instrument.notes:
        notes.append({
            'pitch': note.pitch,
            'start': note.start,
            'end': note.end
        })

    
    # Now use music21 for tab generation.  Important for correct fretboard representations
    try:
        score = music21.converter.parse(midi_file)
    except Exception as e:
        print(f"Error parsing MIDI file with music21: {e}")
        return False

    # Basic tab generation (needs improvement)
    with open(output_file, 'w') as f:
      for part in score.parts:
          for note in part.flat.notes:
              if isinstance(note, music21.note.Note):
                f.write(f"{note.pitch.ps} ")  # Basic pitch output

    print(f"Guitar tab output saved to {output_file}")
    return True




# Example usage:
midi_file_path = "input.mid" # Replace with your MIDI file path.
if midi_to_tab(midi_file_path):
  print("Conversion successful.")
else:
  print("Conversion failed.")