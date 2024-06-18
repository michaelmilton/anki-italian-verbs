from typing import List
from genanki import Model, Note, Deck, Package
from create_content import FlashCardPair

OUTPUT_FILENAME = "output.apkg"

CSS = """.card {
 font-family: arial;
 font-size: 20px;
 text-align: center;
 color: black;
 background-color: white;
}

.cloze {
 font-weight: bold;
 color: blue;
}
.nightMode .cloze {
 color: lightblue;
}
"""

MY_CLOZE_MODEL = Model(
    998877661,
    "My Cloze Model",
    fields=[
        {"name": "Text"},
        {"name": "Extra"},
    ],
    templates=[
        {
            "name": "My Cloze Card",
            "qfmt": "{{cloze:Text}}",
            "afmt": "{{cloze:Text}}<br>{{Extra}}",
        },
    ],
    css=CSS,
    model_type=Model.CLOZE,
)


def create_note(flash_card_pair: FlashCardPair) -> Note:
    """
    Create an Anki note from a FlashCardPair. Assumes we're creating cloze deletions.

    Args:
        flash_card_pair (FlashCardPair): The FlashCardPair to convert to a note.

    Returns:
        Note: The Anki note.
    """
    my_note = Note(
        model=MY_CLOZE_MODEL, fields=[flash_card_pair.cloze, flash_card_pair.extra]
    )
    return my_note


def write_deck(notes: List[Note]) -> None:
    my_deck = Deck(2059400110, "Italian verbs")
    for note in notes:
        my_deck.add_note(note)
    Package(my_deck).write_to_file(OUTPUT_FILENAME)
