"""
Module to execute the commands to generate flash cards and write them to 
an Anki package file.
"""

from random import choice, shuffle
import concurrent.futures
from typing import List
from genanki import Note
from create_content import VerbPackage, VerbPackageList, create_flashcard_pair, NoteList
from create_deck import create_note, write_deck

MAX_WORKERS = 20


def read_data(file_name) -> List[str]:
    """Read data from a file

    Args:
        file_name (_type_): One of the vocab info lists.

    Returns:
        List[str]: A list of the relevant vocab data.
    """
    lines = []

    with open(file_name, "r") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    return lines


key_verbs = read_data("content/key_verbs.txt")
irregular_verbs = read_data("content/irregular_verbs.txt")
regular_verbs = read_data("content/regular_verbs.txt")

basic_tenses = read_data("content/basic_tenses.txt")
advanced_tenses = read_data("content/advanced_tenses.txt")

subjects = read_data("content/subjects.txt")


def get_subject() -> str:
    """
    Fetch a random subject from the file of the list of subjects.

    Returns:
        str: A random subject
    """
    return choice(subjects)


def build_verb_package_list(
    name,
    verbs: List[str] = key_verbs,
    tenses: List[str] = basic_tenses,
    persons: List[str] = read_data("content/persons.txt"),
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
                verb_package = VerbPackage(
                    verb=verb, tense=tense, person=person, subject=get_subject()
                )
                verb_packages.append(verb_package)

    return VerbPackageList(name=name, verb_packages=verb_packages)


def build_note_list(verb_packages: VerbPackageList) -> NoteList:
    pairs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pairs = list(executor.map(create_flashcard_pair, verb_packages.verb_packages))

    notes = []
    for pair in pairs:
        notes.append(create_note(pair))

    shuffle(notes)
    return NoteList(name=verb_packages.name, notes=notes)


if __name__ == "__main__":
    verb_package_lists = [
        build_verb_package_list(
            verbs=key_verbs, tenses=basic_tenses, name="Italian key verbs, basic tenses"
        ),
        build_verb_package_list(
            verbs=key_verbs,
            tenses=advanced_tenses,
            name="Italian key verbs, advanced tenses",
        ),
        build_verb_package_list(
            verbs=regular_verbs,
            tenses=basic_tenses,
            name="Italian regular verbs, basic tenses",
        ),
        build_verb_package_list(
            verbs=irregular_verbs,
            tenses=basic_tenses,
            name="Italian irregular verbs, basic tenses",
        ),
        build_verb_package_list(
            verbs=regular_verbs,
            tenses=advanced_tenses,
            name="Italian regular verbs, advanced tenses",
        ),
        build_verb_package_list(
            verbs=irregular_verbs,
            tenses=advanced_tenses,
            name="Italian irregular verbs, tenses",
        ),
    ]

    for verb_package_list in verb_package_lists:
        write_deck(build_note_list(verb_package_list))
