"""
Module to execute the commands to generate flash cards and write them to
an Anki package file.
"""

from random import choice, shuffle
from typing import List
from constants import (
    key_verbs,
    irregular_verbs,
    regular_verbs,
    basic_tenses,
    advanced_tenses,
    all_persons,
)
from create_deck import write_deck, build_note_list, build_verb_package_list


def go() -> None:
    # Main execution
    verb_package_lists_to_build = [
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
