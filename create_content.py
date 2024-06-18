"""
Code to build the deck
"""

import csv
import os
import openai
from openai import OpenAI
from dataclasses import dataclass
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@dataclass
class VerbPackage:
    verb: str
    tense: str
    person: str
    subject: str


@dataclass
class FlashCardPair:
    cloze: str
    extra: str


FILENAME = "cards.csv"

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI()

VERB = "avere"
TENSE = "Pluperfect Subjunctive"
PERSON = "1st person plural"
SUBJECT = "Roman history"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_flashcard_cloze(verb_package: VerbPackage, sentence):
    create_flashcard_cloze_prompt = f"""
        Look at the following sentence:
        {sentence}
        
        Return the sentence as cloze deletion of the verb {verb_package.verb} in {verb_package.tense} in the sentence {sentence}. 
        Only do the cloze deletion of verb {verb_package.verb} in {verb_package.tense}. Leave any other verbs alone.

        It should look like this:
            Io e i miei amici {{{{c1::siamo}}}} (essere) appassionati di cucina italiana.
            Siamo contenti che noi {{{{c1::abbiamo finito}}}} (finire) di leggere il De Bello Gallico per capire meglio la strategia militare romana.
            {{{{c1::Ho}}}} già {{{{c1::mangiato}}}}. (mangiare)

        The infinitive should always be present in parentheses.

        Where possible the statement should be true. When the tense is in the first or second person, emphasize subjective experiences one might have today of {verb_package.subject}.
        
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
                "content": f"{create_flashcard_cloze_prompt}",
            },
        ],
    )
    return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_flashcard_extra(verb_package: VerbPackage, flashcard_front):
    create_flashcard_answer_prompt = f"""
        Create the extra material for an Anki flashcard. 
        Use HTML format. Do not use Markdown.
        This is the front of the flashcard:
            {flashcard_front}
        For the extra material, first give the English translation of the sentence. 
        Next, give the conjugation of {verb_package.verb} in the {verb_package.tense} tense. Have one conjugated verb per line.
        At the bottom of the response card, there should be two links:
            One link points to a ChatGPT query about the specific topic described in the card, {verb_package.subject}. For example, if the topic is Bolognese lasagna, the query should ask ChatGPT to say more about Bolognese lasagna.
            The other link should point to the relevant section of the relevant Wikipedia page about {verb_package.subject}.
        Only include what is listed above, do not include any other commentary. 

        Here is an example of how the extra card should look.

        <div>
        <p><strong>Traduzione in inglese:</strong></p>
        <p>Opera is a cornerstone of Italian musical culture since the 17th century.</p>
        
        <p><strong>Coniugazione di "essere" nel Presente Indicativo:</strong></p>
        <ul>
            <li>io sono</li>
            <li>tu sei</li>
            <li>lui/lei è</li>
            <li>noi siamo</li>
            <li>voi siete</li>
            <li>loro sono</li>
        </ul>
        
        <p><a href="https://chat.openai.com/?query=Tell%20me%20more%20about%20the%20history%20of%20opera%20in%20Italy">Chiedi a ChatGPT di più sulla storia dell'opera in Italia</a></p>
        <p><a href="https://it.wikipedia.org/wiki/Storia_dell%27opera">Leggi di più su Wikipedia: Storia dell'opera in Italia</a></p>
        </div> 
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
    print(f"Evaluating: {verb_package}")
    sample_sentence = create_italian_sentence(verb_package)
    flashcard_cloze = create_flashcard_cloze(verb_package, sample_sentence)
    flashcard_extra = create_flashcard_extra(verb_package, flashcard_cloze)
    return FlashCardPair(cloze=flashcard_cloze, extra=flashcard_extra)
