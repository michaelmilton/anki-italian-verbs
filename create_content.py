"""
Code to send verb data to OpenAI and get back flashcard content.
"""

import os
import re
import boto3
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from utils import (
    get_conjugated,
    get_conjugation_from_disk,
    get_wikipedia_link_for_subject,
)
from data_types import VerbPackage

WAIT_MIN = 1
WAIT_MAX = 120
STOP_AFTER = 10
CLIENT = openai.OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

EXPENSIVE_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-3.5-turbo"


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
        f"The sentence will be on the topic of {verb_package.subject}."
        if verb_package.person
        in {
            "3rd person singular",
            "3rd person plural",
        }
        else ""
    )
    print(
        f"Requesting from OpenAI sentence for {verb_package.verb_conjugated}. {subject_statement}"
    )
    create_italian_sentence_prompt = f"""
        Create a sentence using this verb:

        {verb_package.verb_conjugated}

        This is the {verb_package.person} {verb_package.tense} of {verb_package.verb}.

        Wrap "{verb_package.verb_conjugated}" with cloze deletion syntax from Anki in a single cloze deletion. 
        Only include the Italian sentence in your response.

        In the context of the sentence, the cloze should look like this:
        {{{{c1::{verb_package.verb_conjugated}}}}}

        {"Do not combine the verb with a different past participle." if verb_package.verb in ("avere", "essere", "stare") else ""}

        {subject_statement}
        
        Include all pronouns, do not skip them.
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
    cloze = f"""
        {verb_package.sentence}
        <p><strong>{verb_package.verb}</strong>
        <p>{verb_package.person}</p><p>{verb_package.tense}</p>
    """
    verb_package.flashcard_cloze = cloze
    return verb_package


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def create_flashcard_extra(verb_package: VerbPackage) -> VerbPackage:
    """
    Creates text for an extra field in the Anki note.
    Calls AWS to get a translation.

    Args:
        verb_package (VerbPackage): The VerbPackage used to create a sentence.

    Returns:
        str: The extra content
    """

    print(f"Requesting translation of sentence for {verb_package.verb_conjugated}")

    client = boto3.client("translate")

    # Remove Anki cloze deletion syntax
    cleaned_sentence = re.sub(r"{{c\d+::|}}", "", verb_package.sentence)
    # Remove any double spaces
    cleaned_sentence = re.sub(r"\s{2,}", " ", cleaned_sentence)

    translation = client.translate_text(
        Text=cleaned_sentence, SourceLanguageCode="it", TargetLanguageCode="en"
    )["TranslatedText"]

    final_output = f"""
            <p><strong>Traduzione in inglese:</strong> {translation}</p>
            {get_conjugation_from_disk(verb_package)}
            {get_wikipedia_link_for_subject(verb_package)}
        """
    verb_package.flashcard_extra = final_output
    return verb_package


def create_flashcard_pair(verb_package: VerbPackage) -> VerbPackage:
    """
    Returns a finalized from a VerbPackage containing the cloze and extra,
    ready for conversion into an Anki note.

    Args:
        verb_package (VerbPackage): The VerbPackage being used to generate the note

    Returns:
        FlashCardPair: The content ready for conversion into a Note.
    """
    verb_package = get_conjugated(verb_package)
    verb_package = create_italian_sentence(verb_package)
    verb_package = create_flashcard_extra(verb_package)
    return verb_package


if __name__ == "__main__":
    pass
