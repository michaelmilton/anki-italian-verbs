"""
Code to build the deck
"""

import csv
import os
import openai
from openai import OpenAI
from dataclasses import dataclass
from typing import List


@dataclass
class VerbPackage:
    verb: str
    tense: str
    person: str
    subject: str


FILENAME = "cards.csv"

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI()

VERB = "avere"
TENSE = "Pluperfect Subjunctive"
PERSON = "1st person plural"
SUBJECT = "Roman history"


def create_italian_sentence(verb_package: VerbPackage) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an Italian teacher. You create sentences in Italian using specified verbs and tenses. You make interesting, nontrivial statements of historical, technical, and literary fact based on information from Wikipedia.",
            },
            {
                "role": "user",
                "content": f"Give me a sentence using {verb_package.verb} in the {verb_package.person} {verb_package.tense} tense. The sentence will be on the subject of {verb_package.subject}. Do not provide the English translation or any other information besides the Italian sentence",
            },
        ],
    )
    return response.choices[0].message.content


def create_flashcard_question(verb_package: VerbPackage, sentence):
    create_flashcard_question_prompt = f"""
        Look at the following sentence:
        {sentence}
        
        Return the sentence as cloze deletion of the verb {verb_package.verb} in the sentence {sentence}. 

        It should look like this:
            Io e i miei amici {{{{c1::siamo}}}} (essere) appassionati di cucina italiana.
            Siamo contenti che noi {{{{c1::abbiamo finito}}}} (finire) di leggere il De Bello Gallico per capire meglio la strategia militare romana.
            {{{{c1::Ho}}}} già {{{{c1::mangiato}}}}. (mangiare)

        Only return the cloze deletion. Do not give any other commentary.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You create Anki cards for language learning",
            },
            {
                "role": "user",
                "content": f"{create_flashcard_question_prompt}",
            },
        ],
    )
    return response.choices[0].message.content


def create_flashcard_answer(verb_package: VerbPackage, flashcard_front):
    create_flashcard_answer_prompt = f"""
        Create the answer for an Anki flashcard. This is the front of the flashcard:
            {flashcard_front}
        The first line of the answer should be the content of c1 field, which is the 
        conjugation of the verb {verb_package.verb} with the tense {verb_package.tense} in the person, {verb_package.person}. 
        Next, give an explanation in Italian of the grammar as used on the front of the card.
        Next, give the conjugation of {verb_package.verb} in the {verb_package.tense} tense. 
        At the bottom of the response card, there should be two links:
            One link points to a ChatGPT query about the specific topic described in the card, {verb_package.subject}. For example, if the topic is Bolognese lasagna, the query should ask ChatGPT to say more about Bolognese lasagna.
            The other link should point to the relevant section of the relevant Wikipedia page about {verb_package.subject}.
        Only include what is listed above, do not include any other commentary. 
    """
    response = client.chat.completions.create(
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
    return response.choices[0].message.content


def create_flashcard_pair(verb_package: VerbPackage):
    sample_sentence = create_italian_sentence(verb_package)
    flashcard_question = create_flashcard_question(verb_package, sample_sentence)
    # print(flashcard_question)
    flashcard_answer = create_flashcard_answer(verb_package, flashcard_question)
    # print(flashcard_answer)
    return {"question": flashcard_question, "answer": flashcard_answer}


def write_flashcard_pairs(pairs: List):
    field_names = pairs[0].keys()
    with open(FILENAME, mode="w", newline="", encoding="UTF-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names, delimiter="\t")

        # Write the header
        writer.writeheader()

        # Write the data
        for row in pairs:
            writer.writerow(row)

    print(f"Data has been written to {FILENAME}")


my_verb_package = VerbPackage(
    verb="avere",
    tense="Pluperfect Subjunctive",
    person="1st person plural",
    subject="Roman history",
)
write_flashcard_pairs([create_flashcard_pair(my_verb_package)])
