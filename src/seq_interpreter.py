import music21 as m21
import numpy as np
import json


class Main:
    def __init__(self, parser):
        with open("../Data/token_to_chord.json", "r") as fp:
            self.token_to_chord = json.load(fp)
        # Convert the dictionary keys to integers
        self.token_to_chord = {int(k): v for k, v in self.token_to_chord.items()}

        self.VOCAB_SIZE = len(self.token_to_chord) + 2
        self.parser = parser

        with open("../Data/chord_map.json", "r") as fp:
            chord_map = json.load(fp)

        # Construct a map of notes from a chord
        all_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.chord_to_notes = {}
        for chord, chord_notes in chord_map.items():
            root, ext = self.parser.get_root_and_ext(chord)
            ext = self.parser.update_if_slash(root, ext)

            chord_notes = np.array(chord_notes)
            chord_notes -= chord_notes.min()

            for i, note in enumerate(all_notes):
                self.chord_to_notes[note + ext] = chord_notes + i

    def token_seq_to_chords(self, seq):
        """Convert a token sequence to a string of chord symbols separated by |"""
        chords = []
        for i in seq:
            if i == self.VOCAB_SIZE - 1:
                break
            if i != self.VOCAB_SIZE - 2:
                possible_chords = self.token_to_chord[i.item()]
                # Use the shortest chord symbol
                chords.append(min(possible_chords, key=len))
        return " | ".join(chords)

    def play_seq(self, seq):
        """Play a sequence of tokens using music21"""
        s = m21.stream.Stream()
        for chord in seq.tolist():
            if chord == self.VOCAB_SIZE - 1:  # End of sequence
                break
            if chord == self.VOCAB_SIZE - 2:  # Start of sequence
                continue

            root, ext = self.parser.get_root_and_ext(
                min(self.token_to_chord[chord], key=len).replace(" ", "")
            )
            ext = self.parser.update_if_slash(root, ext)
            notes = self.chord_to_notes[root + ext]
            chord = m21.chord.Chord((notes + 60).tolist())
            chord.duration.quarterLength = 2
            s.append(chord)
        s.show("midi")
