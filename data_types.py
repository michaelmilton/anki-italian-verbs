from dataclasses import dataclass
from random import choice
from typing import List
from genanki import Note
from verb_data import subjects


@dataclass
class VerbPackage:
    """
    This dataclass encapsulates the data necessary to create a single
    cloze deletion flashcard. It describes the verb we want to study, the
    tense and person to learn, and the subject that we want the cloze
    deletion sentence to describe.

    It also accumulates the cloze and extra values that get fed into
    the functions from `create_deck` that create the Anki notes and
    package.

    This object gets mutated throughout the process of developing a note.
    """

    verb: str
    tense: str
    person: str
    subject: str = choice(subjects)
    sentence: str = ""
    flashcard_cloze: str = ""
    flashcard_extra: str = ""
    verb_conjugated: str = ""


@dataclass
class VerbPackageList:
    """
    Encapsulates a list of VerbPackages
    """

    name: str
    verb_packages: List[VerbPackage]


@dataclass
class NoteList:
    """
    Encapsulates a list of Notes
    """

    name: str
    notes: List[Note]
