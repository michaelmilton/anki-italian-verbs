"""
Module to take genai-created content and package it into Anki material.
"""

from random import shuffle
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from genanki import Model, Note, Deck, Package
from create_content import create_flashcard_pair
from verb_data import (
    key_verbs,
    irregular_verbs,
    regular_verbs,
    basic_tenses,
    advanced_tenses,
    all_persons,
)
from data_types import NoteList, VerbPackage, VerbPackageList


OUTPUT_FILENAME = "output.apkg"

MAX_WORKERS = 20

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


def create_note(verb_package: VerbPackage) -> Note:
    """
    Create an Anki note from a VerbPackage. Assumes we're creating cloze deletions.

    Args:
        verb_package (VerbPackage): The VerbPackage to convert to a note.

    Returns:
        Note: The Anki note.
    """
    my_note = Note(
        model=MY_CLOZE_MODEL,
        fields=[verb_package.flashcard_cloze, verb_package.flashcard_extra],
    )
    return my_note


def write_deck(note_list: NoteList) -> None:
    """
    From a NoteList, write to disk an Anki deck package file.

    Args:
        notes (List[Note]): The list of notes
    """
    my_deck = Deck(int(time.time()), note_list.name)
    for note in note_list.notes:
        my_deck.add_note(note)
    Package(my_deck).write_to_file(note_list.name + ".apkg")


def build_note_list(verb_package_list: VerbPackageList) -> NoteList:
    """
    From a list of verb packages, build a list of notes.

    Args:
        verb_package_list (VerbPackageList): The verb packages.

    Returns:
        NoteList: The note list.
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        final_verb_packages = list(
            executor.map(create_flashcard_pair, verb_package_list.verb_packages)
        )

    notes = []
    for final_verb_package in final_verb_packages:
        notes.append(create_note(final_verb_package))

    shuffle(notes)
    return NoteList(name=verb_package_list.name, notes=notes)


def build_verb_package_list(
    name,
    verbs: List[str] = key_verbs,
    tenses: List[str] = basic_tenses,
    persons: List[str] = all_persons,
) -> VerbPackageList:
    """
    Build a list of VerbPackage objects given the requested collection of verbs and tenses.

    Args:
        verbs (List[str], optional): The verb set to use. Defaults to key_verbs.
        tenses (List[str], optional): The tenses to use. Defaults to basic_tenses.
        persons (List[str], optional): The persons to use. Defaults to persons.

    Returns:
        List[VerbPackage]: VerbPackage objects for which we will make Anki notes.
    """
    verb_packages = []
    for verb in verbs:
        for tense in tenses:
            for person in persons:
                verb_package = VerbPackage(verb=verb, tense=tense, person=person)
                verb_packages.append(verb_package)

    return VerbPackageList(name=name, verb_packages=verb_packages)
