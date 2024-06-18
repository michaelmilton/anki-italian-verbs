from create_content import VerbPackage, create_flashcard_pair
from create_deck import create_note, write_deck
from random import choice, shuffle
import concurrent.futures


def read_data(file_name):
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
persons = read_data("content/persons.txt")


def get_subject() -> str:
    return choice(subjects)


def build_note_list(verbs=key_verbs, tenses=basic_tenses, persons=persons):
    verb_packages = []
    for verb in verbs:
        for tense in tenses:
            for person in persons:
                verb_package = VerbPackage(
                    verb=verb, tense=tense, person=person, subject=get_subject()
                )

                verb_packages.append(verb_package)

    pairs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        pairs = list(executor.map(create_flashcard_pair, verb_packages))

    notes = []
    for pair in pairs:
        notes.append(create_note(pair))

    shuffle(notes)
    return notes


def build_all_notes():
    notes = []

    notes.extend(build_note_list(verbs=key_verbs, tenses=basic_tenses))
    notes.extend(build_note_list(verbs=key_verbs, tenses=advanced_tenses))
    notes.extend(build_note_list(verbs=regular_verbs, tenses=basic_tenses))
    notes.extend(build_note_list(verbs=irregular_verbs, tenses=basic_tenses))
    notes.extend(build_note_list(verbs=regular_verbs, tenses=advanced_tenses))
    notes.extend(build_note_list(verbs=irregular_verbs, tenses=advanced_tenses))

    return notes


write_deck(build_all_notes())
