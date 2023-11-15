import music21 as m21
import re


class Main:
    def __init__(self):
        # Define notes
        self.notes = "C- C C# D- D D# E- E E# F- F F# G- G G# A- A A# B- B B#".split(
            " "
        )

        # Define a map for slash chords
        self.slash_map = {}
        for root in self.notes:
            for bass in self.notes:
                figure = m21.roman.romanNumeralFromChord(
                    m21.chord.Chord(bass), root
                ).figure
                # Upper the figure except for the bbb, bb, b, #, ##, ### in the beggining (if there is one)
                match = re.match(r"^(bbb|bb|b|#|##|###)?(.*)", figure)
                if match:
                    accidentals, numeral = match.groups()
                    if not accidentals:
                        accidentals = ""
                    figure = accidentals + numeral.upper()

                self.slash_map[f"{root}/{bass}"] = figure

    def update_if_slash(self, root, extension):
        """If the extension is a slash chord, get it as a roman numeral."""
        if "/" in extension:
            bass = extension.split("/")[1].replace("b", "-")
            slash_pair = root.replace("b", "-") + "/" + bass
            if slash_pair in self.slash_map:
                extension = extension.split("/")[0] + "/" + self.slash_map[slash_pair]
        return extension

    def inverse_update_if_slash(self, root, extension):
        """Revert slash chord from roman numeral back to note form."""

        if "/" in extension:
            roman_numeral = extension.split("/")[1]
            root_formatted = root.replace("b", "-")

            # Find the matching bass note for the roman numeral
            for slash_pair, figure in self.slash_map.items():
                slash_root, slash_bass = slash_pair.split("/")
                if figure == roman_numeral and slash_root == root_formatted:
                    bass_note = slash_bass.replace("-", "b")
                    extension = extension.split("/")[0] + "/" + bass_note
                    break
        return extension

    def get_root_and_ext(self, ch):
        """Get the root and extension of a chord."""
        if len(ch) > 2 and (ch[1:3] == "##" or ch[1:3] == "bb"):
            return ch[:3].replace("b", "-"), ch[3:]
        if len(ch) > 1 and (ch[1] == "#" or ch[1] == "b"):
            return ch[:2].replace("b", "-"), ch[2:]
        return ch[:1], ch[1:]

    def get_root(self, ch):
        """Get the root of a chord."""
        if len(ch) > 2 and (ch[1:3] == "##" or ch[1:3] == "bb"):
            return ch[:3].replace("b", "-")
        if len(ch) > 1 and (ch[1] == "#" or ch[1] == "b"):
            return ch[:2].replace("b", "-")
        return ch[:1]

    def get_extension(self, ch):
        """Get the extension of a chord."""
        if len(ch) > 2 and (ch[1:3] == "##" or ch[1:3] == "bb"):
            return ch[3:]
        if len(ch) > 1 and (ch[1] == "#" or ch[1] == "b"):
            return ch[2:]
        return ch[1:]
