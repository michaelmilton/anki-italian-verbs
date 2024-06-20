"""
Module to execute the commands to generate flash cards and write them to
an Anki package file.
"""

from random import choice, shuffle
import concurrent.futures
from typing import List
from create_content import (
    VerbPackage,
    VerbPackageList,
    create_flashcard_pair,
    NoteList,
    get_conjugated,
    key_verbs,
    irregular_verbs,
    regular_verbs,
    basic_tenses,
    advanced_tenses,
    persons,
)
from create_deck import create_note, write_deck

MAX_WORKERS = 20


def build_verb_package_list(
    name,
    verbs: List[str] = key_verbs,
    tenses: List[str] = basic_tenses,
    persons: List[str] = persons,
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


def build_note_list(verb_package_list: VerbPackageList) -> NoteList:
    """
    From a list of verb packages, build a list of notes.

    Args:
        verb_package_list (VerbPackageList): The verb packages.

    Returns:
        NoteList: The note list.
    """
    pairs = []
    print(
        f"""
        In `build_note_list`. The type of the `verb_package_list` is {type(verb_package_list)}.
        The type of `verb_package_list.verb_packages` is {type(verb_package_list.verb_packages)}
        """
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        final_verb_packages = list(
            executor.map(create_flashcard_pair, verb_package_list.verb_packages)
        )

    notes = []
    for final_verb_package in final_verb_packages:
        notes.append(create_note(final_verb_package))

    shuffle(notes)
    return NoteList(name=verb_package_list.name, notes=notes)


def go() -> None:
    # Main execution
    verb_package_lists_to_build = [
        # build_verb_package_list(
        #     verbs=key_verbs, tenses=basic_tenses, name="Italian key verbs, basic tenses"
        # ),
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
        # build_verb_package_list(
        #     verbs=irregular_verbs,
        #     tenses=basic_tenses,
        #     name="Italian irregular verbs, basic tenses",
        # ),
        # build_verb_package_list(
        #     verbs=regular_verbs,
        #     tenses=advanced_tenses,
        #     name="Italian regular verbs, advanced tenses",
        # ),
        # build_verb_package_list(
        #     verbs=irregular_verbs,
        #     tenses=advanced_tenses,
        #     name="Italian irregular verbs, tenses",
        # ),
    ]

    for verb_package_list_to_build in verb_package_lists_to_build:
        write_deck(build_note_list(verb_package_list_to_build))


if __name__ == "__main__":
    go()
