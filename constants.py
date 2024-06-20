import os
from typing import List
import openai


def read_data(file_name: str) -> List[str]:
    """Read data from a file

    Args:
        file_name (_type_): One of the vocab info lists.

    Returns:
        List[str]: A list of the relevant vocab data.
    """
    lines = []

    with open(file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    return lines


subjects = read_data("content/subjects.txt")
key_verbs = read_data("content/key_verbs.txt")
irregular_verbs = read_data("content/irregular_verbs.txt")
regular_verbs = read_data("content/regular_verbs.txt")
basic_tenses = read_data("content/basic_tenses.txt")
advanced_tenses = read_data("content/advanced_tenses.txt")
all_persons = read_data("content/persons.txt")

WAIT_MIN = 1
WAIT_MAX = 120
STOP_AFTER = 10

CLIENT = openai.OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

EXPENSIVE_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-3.5-turbo"
