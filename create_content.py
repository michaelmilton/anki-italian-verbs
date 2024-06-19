"""
Code to send verb data to OpenAI and get back flashcard content.
"""

import os
import json
from dataclasses import dataclass
from typing import List
import boto3
from genanki import Note
import openai
from openai import OpenAI
import spacy
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

EXPENSIVE_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-3.5-turbo"

nlp = spacy.load("it_core_news_lg")


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
    sentence: str = ""
    flashcard_cloze: str = ""
    flashcard_extra: str = ""


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


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def create_italian_sentence(verb_package: VerbPackage) -> VerbPackage:
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
        Do not combine {verb_package.verb} with the past participle of another verb.
    """
    response = CLIENT.chat.completions.create(
        model=EXPENSIVE_MODEL,
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
    verb_package.sentence = str(response.choices[0].message.content)
    return verb_package


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def create_flashcard_cloze_old(verb_package: VerbPackage) -> VerbPackage:
    """
    Takes the generated sentence and converts it into a cloze deletion.

    Args:
        verb_package (VerbPackage): The VerbPackage that was used in creating the sentence.

    Returns:
        str: The cloze sentence
    """
    create_flashcard_cloze_prompt = f"""
        Look at the following sentence:
        {verb_package.sentence}
        
        Return the sentence as cloze deletion of the verb {verb_package.verb} in {verb_package.tense}. 
        Leave any other verbs alone.

        The output should look like this:
            <p>Io e i miei amici {{{{c1::siamo}}}} appassionati di cucina italiana.</p>
            
            <p>Siamo contenti che noi {{{{c1::abbiamo finito}}}} di leggere il De Bello Gallico per capire meglio la strategia militare romana.</p>
            
            <p>{{{{c1::Ho}}}} già {{{{c1::mangiato}}}}.</p>

            <p>{{{{c1::Sei stato}}}} immerso nella complessità della vita e delle opere di Dante Alighieri.</p>

        Only return the cloze sentence, don't return any other commentary.
        If the conjugation of {verb_package.verb} is compound, reflexive or both, all auxiliary parts of the verb must be included in the cloze. 
        This is incorrect: "Siamo contenti che noi abbiamo {{{{c1::finito}}}} di leggere."
        This is correct: "Siamo contenti che noi {{{{c1::abbiamo finito}}}} di leggere."
    """
    response = CLIENT.chat.completions.create(
        model=EXPENSIVE_MODEL,
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
    final_output = f"""
            <p>{response.choices[0].message.content} ({verb_package.verb})</p>
            <p>{verb_package.verb} {verb_package.person} {verb_package.tense}</p>
        """
    verb_package.flashcard_cloze = final_output
    return verb_package


def create_cloze(verb_package: VerbPackage) -> VerbPackage:
    """
    Spacy-powered function to create a cloze sentence from a sentence.

    Args:
        verb_package (VerbPackage): _description_

    Returns:
        VerbPackage: _description_
    """
    doc = nlp(verb_package.sentence)
    infinitive = verb_package.verb
    cloze_text = []
    verb_phrases = []
    current_phrase = []

    for token in doc:
        if token.lemma_ == infinitive and token.pos_ in {"AUX", "VERB"}:
            current_phrase.append(token.text)
        else:
            if current_phrase:
                verb_phrases.append(current_phrase)
                current_phrase = []
            cloze_text.append(token.text)

    if current_phrase:
        verb_phrases.append(current_phrase)

    cloze_text_with_phrases = []
    phrase_idx = 0

    for token in doc:
        if token.lemma_ == infinitive and token.pos_ in {"AUX", "VERB"}:
            if token.text in verb_phrases[phrase_idx]:
                if not current_phrase:
                    cloze_text_with_phrases.append(f"{{{{c1::{token.text}")
                else:
                    cloze_text_with_phrases.append(token.text)
                current_phrase.append(token.text)
                if len(current_phrase) == len(verb_phrases[phrase_idx]):
                    cloze_text_with_phrases.append("}}")
                    phrase_idx += 1
                    current_phrase = []
            else:
                cloze_text_with_phrases.append(token.text)
        else:
            cloze_text_with_phrases.append(token.text)

    verb_package.flashcard_cloze = " ".join(cloze_text_with_phrases)
    return verb_package


def create_flashcard_cloze(verb_package: VerbPackage) -> VerbPackage:
    """
    Create the cloze and add hint material.

    Args:
        verb_package (VerbPackage): The verb_package

    Returns:
        VerbPackage: The verb_package, now with updated cloze
    """
    verb_package = create_cloze(verb_package)
    final_output = f"""
            <p>{verb_package.flashcard_cloze}</p>
            <p>{verb_package.verb} {verb_package.person} {verb_package.tense}</p>
        """
    verb_package.flashcard_cloze = final_output
    return verb_package


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_flashcard_extra(verb_package: VerbPackage) -> VerbPackage:
    """
    Creates text for an extra field in the Anki note.

    Args:
        verb_package (VerbPackage): The VerbPackage used to create a sentence.

    Returns:
        str: The extra content
    """
    client = boto3.client("translate")

    translation = client.translate_text(
        Text=verb_package.sentence, SourceLanguageCode="it", TargetLanguageCode="en"
    )["TranslatedText"]
    final_output = f"""
            <p><strong>Traduzione in inglese:</strong></p> 
            <p>{translation}</p>
            {get_conjugation_from_disk(verb_package)}
            {get_wikipedia_link_for_subject(verb_package)}
        """
    verb_package.flashcard_extra = final_output
    return verb_package


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
    verb_tense = verb_package.verb + " " + verb_package.tense

    if os.path.exists(CONJUGATIONS_FILE):
        with open(CONJUGATIONS_FILE, "r", encoding="utf-8") as json_file:
            conjugations = json.load(json_file)
        try:
            conjugation = conjugations[verb_tense]
        except (
            KeyError
        ):  # If this conjugation hasn't been cached, look it up and persist it
            conjugation = get_conjugation_from_llm(verb_package)
            conjugations[verb_tense] = conjugation
            with open(CONJUGATIONS_FILE, "w", encoding="utf-8") as json_file:
                json.dump(conjugations, json_file)
    else:
        conjugation = get_conjugation_from_llm(verb_package)
        conjugations[verb_package.verb] = {verb_package.tense: conjugation}
        with open(CONJUGATIONS_FILE, "w", encoding="utf-8") as json_file:
            json.dump(conjugations, json_file)
    return conjugation


def get_wikipedia_link_for_subject(verb_package: VerbPackage) -> str:
    """
    Returns a wikipedia link to search for this topic.

    Args:
        verb_package (VerbPackage): The package containing the subject.

    Returns:
        str: The HTML containing the link for the extra card.
    """

    base_url = "https://en.wikipedia.org/w/index.php?fulltext=1&search="
    subject = verb_package.subject.replace(" ", "_")
    final_url = base_url + subject

    final_output = f"""
            <p>For more on this topic see 
            <a href "{final_url}" target="_blank">Wikipedia</a> </p>
        """

    return final_output


def get_conjugation_from_llm(verb_package: VerbPackage) -> str:
    """
    In the absence of a cached conjugation, look one up from the LLM.

    Args:
        verb_package (VerbPackage): The package containing the verb and tense.

    Returns:
        str: The conjugation
    """

    response = CLIENT.chat.completions.create(
        model=EXPENSIVE_MODEL,
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
                     p   <li>voi siete</li>
                        <li>loro sono</li>
                    </ul>
                    Only provide the conjugated verbs, no other commentary.
                """,
            },
        ],
    )
    final_output = f"""
            <p><strong>Coniugazione di "{verb_package.verb}" nel {verb_package.tense}:</strong></p>
            <p>{response.choices[0].message.content}</p>

        """
    return final_output


def create_flashcard_pair(verb_package: VerbPackage) -> VerbPackage:
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
    verb_package = create_italian_sentence(verb_package)
    verb_package = create_flashcard_cloze(verb_package)
    verb_package = create_flashcard_extra(verb_package)
    return verb_package


if __name__ == "__main__":
    pass
