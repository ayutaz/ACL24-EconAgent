import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
from dotenv import load_dotenv

load_dotenv()

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1m, completion_cost_1m = 0.40, 1.60

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    from functools import partial
    get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum([cost for _, cost in results])
    return [response for response, _ in results], total_cost

def get_completion(dialogs, temperature=0, max_tokens=100):
    import time
    from openai import OpenAI

    client = OpenAI()  # 環境変数 OPENAI_API_KEY から自動読み込み

    max_retries = 20
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=dialogs,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            this_cost = prompt_tokens / 1_000_000 * prompt_cost_1m + completion_tokens / 1_000_000 * completion_cost_1m
            return response.choices[0].message.content, this_cost
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error", 0

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
