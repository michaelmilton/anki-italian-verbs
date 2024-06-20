import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from constants import EXPENSIVE_MODEL, CLIENT, WAIT_MAX, WAIT_MIN, STOP_AFTER
from data_types import VerbPackage

CONJUGATIONS_FILE = "conjugations.txt"


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
def get_conjugated(verb_package: VerbPackage) -> VerbPackage:
    """
    Return the conjugation of a verb, given an infinitive, person, and tense.

    Args:
        verb_package (VerbPackage): The verb package

    Returns:
        VerbPackage: The verb package with conjugated form added.
    """
    get_conjugated_prompt = f"""
        Return the {verb_package.person} {verb_package.tense} of {verb_package.verb}.
        Only include the Italian conjugated verb. Do not include any other information.
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
                "content": get_conjugated_prompt,
            },
        ],
    )
    verb_package.verb_conjugated = (
        str(response.choices[0].message.content).lower().strip()
    )

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


@retry(
    wait=wait_random_exponential(min=WAIT_MIN, max=WAIT_MAX),
    stop=stop_after_attempt(STOP_AFTER),
)
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
            <a href="{final_url}" target="_blank">Wikipedia</a> </p>
        """

    return final_output
