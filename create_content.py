"""
Code to send verb data to OpenAI and get back flashcard content.
"""

import os
import json
from dataclasses import dataclass
from typing import List
from genanki import Note
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

WAIT_MIN = 1
WAIT_MAX = 120
STOP_AFTER = 10
CLIENT = OpenAI()
CONJUGATIONS_FILE = "conjugations.txt"
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class VerbPackage:
    """
    This dataclass encapsulates the data necessary to create a single
    cloze deletion flashcard. It describes the verb we want to study, the
    tense and person to learn, and the subject that we want the cloze
    deletion sentence to describe.
    """

    verb: str
    tense: str
    person: str
    subject: str


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


@dataclass
class FlashCardPair:
    """
    This dataclass encapsulates the completed output from the genai process,
    namely the cloze deletion and the extra material for the flashcard.
    """

    cloze: str
    extra: str


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def create_italian_sentence(verb_package: VerbPackage) -> str:
    """
    Creates a sentence in Italian from the given VerbPackage.

    Args:
        verb_package (VerbPackage): The VerbPackage to use to create a sentence.

    Returns:
        str: The created sentence.
    """
    subject_statement = (
        f"The sentence will be on the subject of {verb_package.subject}."
        if verb_package.person
        in {
            "1st person singular",
            "2nd person singular",
            "1st person plural",
            "2nd person plural",
        }
        else ""
    )
    create_italian_sentence_prompt = f"""
        Make a sentence using {verb_package.verb} in the {verb_package.person} {verb_package.tense}. 
        {subject_statement} 
        Do not provide the English, only the Italian sentence.
        If {verb_package.verb} is essere, avere, stare, or fare, do not combine it with the past participle of another verb.
    """
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an Italian teacher.",
            },
            {
                "role": "user",
                "content": create_italian_sentence_prompt,
            },
        ],
    )
    return response.choices[0].message.content


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def create_flashcard_cloze(verb_package: VerbPackage, sentence) -> str:
    """
    Takes the generated sentence and converts it into a cloze deletion.

    Args:
        verb_package (VerbPackage): The VerbPackage that was used in creating the sentence.
        sentence (_type_): The sentence to convert to cloze.

    Returns:
        str: The cloze sentence
    """
    create_flashcard_cloze_prompt = f"""
        Look at the following sentence:
        {sentence}
        
        Return the sentence as cloze deletion of the verb {verb_package.verb} in {verb_package.tense}. 
        Leave any other verbs alone.

        The output  should look like this:
            <p>Io e i miei amici {{{{c1::siamo}}}} (essere) appassionati di cucina italiana.</p>
            <p>First person plural, Presente Indicativo</p>
            
            <p>Siamo contenti che noi {{{{c1::abbiamo finito}}}} (finire) di leggere il De Bello Gallico per capire meglio la strategia militare romana.</p>
            <p>First person plural, Passato Prossimo Indicativo</p>
            
            <p>{{{{c1::Ho}}}} già {{{{c1::mangiato}}}}. (mangiare)</p>
            <p>First person singular, Passato Prossimo Indicativo</p>

            <p>{{{{c1::Sei stato}}}} immerso nella complessità della vita e delle opere di Dante Alighieri. (essere)</p>
            <p>Second person singular, Passato Prossimo Indicativo</p>

        If the conjugat in {verb_package.verb} is compound, reflexive or both, all parts must be included in the cloze. 
        This is incorrect: "Siamo contenti che noi abbiamo {{{{c1::finito}}}} (finire) di leggere."
    """
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You create Anki cards for language learning",
            },
            {
                "role": "user",
                "content": f"{create_flashcard_cloze_prompt}",
            },
        ],
    )
    return f"""
        <p>{response.choices[0].message.content} ({verb_package.verb})<\p>
        <p>{verb_package.person} {verb_package.tense}<\p>
    """


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_flashcard_extra(verb_package: VerbPackage, cloze) -> str:
    """
    Creates text for an extra field in the Anki note.

    Args:
        verb_package (VerbPackage): The VerbPackage used to create a sentence.
        cloze (_type_): The cloze sentence

    Returns:
        str: The extra content
    """
    create_flashcard_answer_prompt = f"""
        Create the extra material for an Anki flashcard. 
        This is the content:
            {cloze}
        Give the English translation.
        Do not include any other commentary. 

        Use HTML format. Here is an example of how the extra content should look.

        <p>Opera is a cornerstone of Italian musical culture since the 17th century.</p>

    """
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an Italian teacher who creates the rear response cards of an Anki deck on grammar. You speak only in Italian. You do not speak in English.",
            },
            {
                "role": "user",
                "content": create_flashcard_answer_prompt,
            },
        ],
    )
    return f"""
        <p><strong>Traduzione in inglese:</strong></p> 
        <p>{response.choices[0].message.content}</p>
        {get_conjugation_from_disk(verb_package)}
    """


def get_conjugation_from_disk(verb_package: VerbPackage) -> str:
    """
    Looks to cached conjugation on disk to find a conjugation of a verb in a tense.
    If the conjugation is not cached, it will consult LLM.

    Args:
        verb_package (VerbPackage): The package containing the verb and tense.

    Returns:
        str: The conjugation
    """
    conjugation = ""
    conjugations = {}
    if os.path.exists(CONJUGATIONS_FILE):
        with open(CONJUGATIONS_FILE, "r", encoding="utf-8") as json_file:
            conjugations = json.load(json_file)
        try:
            conjugation = conjugations[verb_package.verb][verb_package.tense]
        except (
            KeyError
        ):  # If this conjugation hasn't been cached, look it up and persist it
            conjugation = get_conjugation_from_llm(verb_package)
            conjugations[verb_package.verb][verb_package.tense] = conjugation
            with open("my_dict.json", "w", encoding="utf-8") as json_file:
                json.dump(conjugations, json_file)
    else:
        pass
    return conjugation


def get_conjugation_from_llm(verb_package: VerbPackage) -> str:
    """
    In the absence of a cached conjugation, look one up from the LLM.

    Args:
        verb_package (VerbPackage): The package containing the verb and tense.

    Returns:
        str: The conjugation
    """

    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an Italian teacher.",
            },
            {
                "role": "user",
                "content": f"""
                    Return the conjugation of the verb {verb_package.verb} in {verb_package.tense}.
                    It should look like this:
                    
                    <ul>
                        <li>io sono</li>
                        <li>tu sei</li>
                        <li>lui/lei è</li>
                        <li>noi siamo</li>
                        <li>voi siete</li>
                        <li>loro sono</li>
                    </ul>
                """,
            },
        ],
    )
    return f"""
        <p><strong>Coniugazione di "essere" nel Presente Indicativo:</strong></p>
        <p>{response.choices[0].message.content}</p>

    """


def create_flashcard_pair(verb_package: VerbPackage) -> FlashCardPair:
    """
    Returns a FlashCardPair from a VerbPackage. The FlashCardPair is ready for
    conversion into an Anki note.

    This function precipitates three API calls to OpenAI.

    Args:
        verb_package (VerbPackage): The VerbPackage being used to generate the note

    Returns:
        FlashCardPair: The content ready for conversion into a Note.
    """
    print(f"Evaluating: {verb_package}")
    sample_sentence = create_italian_sentence(verb_package)
    flashcard_cloze = create_flashcard_cloze(verb_package, sample_sentence)
    flashcard_extra = create_flashcard_extra(verb_package, flashcard_cloze)
    return FlashCardPair(cloze=flashcard_cloze, extra=flashcard_extra)
