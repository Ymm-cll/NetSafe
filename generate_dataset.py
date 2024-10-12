import json

from openai import OpenAI
from tqdm import tqdm
import methods
import random
from datasets import load_dataset

random.seed(42)


def generate_dataset(dataset_name, dataset_path, sample_num=None):
    ds = load_dataset(*dataset_path)
    ds_total = []
    methods.create_directory("./dataset")
    save_path = f"./dataset/{dataset_name}.jsonl"
    for key in ds.keys():
        for item in ds[key]:
            ds_total.append(item)
    if sample_num:
        random.shuffle(ds_total)
        sampled_ds = random.sample(ds_total, sample_num)
    else:
        sampled_ds = ds_total
    with open(save_path, "w", encoding="utf-8") as f:
        for item in sampled_ds:
            f.write(f"{item}\n")


def sample_by_level(dataset, sample_size):
    """
    Function to sample the dataset based on level distribution.
    Prioritize higher levels if the sample size cannot be evenly distributed.

    :param dataset: Dataset to sample from.
    :param sample_size: Total number of samples to retrieve.
    :return: Sampled data as a list of dictionaries, retaining all fields.
    """
    levels = sorted(dataset.unique('level'))
    samples_per_level = sample_size // len(levels)
    remainder = sample_size % len(levels)

    # Sample the required amount from each level
    sampled_data = []
    for level in levels:
        level_data = dataset.filter(lambda example: example['level'] == level)
        if remainder > 0:
            sampled_data.append(level_data.shuffle(seed=42).select(range(samples_per_level + 1)))
            remainder -= 1
        else:
            sampled_data.append(level_data.shuffle(seed=42).select(range(samples_per_level)))

    # Concatenate all sampled data
    return [example for subset in sampled_data for example in subset]


def generate_dataset_math(output_path, sample_size):
    """
    Function to generate the required dataset with specific algebra and geometry samples,
    ensuring that the levels are balanced.

    :param output_path: Path to save the generated dataset.
    """
    # Load the dataset (assuming you need only the "train" split)
    ds = load_dataset("lighteval/MATH", "all")['test']

    # Filter for algebra and geometry categories
    algebra_ds = ds.filter(lambda example: example['type'] == 'Algebra')
    geometry_ds = ds.filter(lambda example: example['type'] == 'Geometry')

    # Sample from algebra and geometry
    algebra_sample = sample_by_level(algebra_ds, sample_size)
    geometry_sample = sample_by_level(geometry_ds, sample_size)

    # Combine both samples
    final_sample = algebra_sample + geometry_sample

    # Save each row as a JSON dictionary in JSON Lines format
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in final_sample:
            f.write(f"{record}\n")


def generate_math_answer(input_path, output_path):
    dataset = methods.get_dataset(input_path)
    client = methods.get_client()

    for i, data in tqdm(enumerate(dataset)):
        problem = data["problem"]
        solution = data["solution"]
        error_task = """
            Based on the solution provided to the problem below, figure out the final answer number:
            Problem: {}
            Solution: {}
            You must response with only the final answer number without any other information.
            """.format(problem, solution)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": error_task}],
            temperature=0,
            max_tokens=128
        ).choices[0].message.content

        # Add the incorrect answer to the original data
        data["answer"] = response
        data["task_id"] = i + 1
        with open(output_path, "a", encoding="utf-8") as f_out:
            # Write the result in JSONL format (each entry in one line)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


def generate_math_error(input_path, output_path):
    dataset = methods.get_dataset(input_path)
    client = methods.get_client()

    for data in tqdm(dataset):
        problem = data["problem"]
        error_task = """
            Please provide a step-by-step solution to the following math problem.
            During the solution process, introduce several types of mistakes, such as using incorrect formulas, performing wrong arithmetic operations, misunderstanding percentages, or making wrong assumptions.
            As the solution progresses, the reasoning should gradually become more chaotic.
            The final answer must be incorrect and a clear result of the earlier missteps.
            Ensure that the errors seem plausible to someone not paying close attention, and don't indicate which steps contain the mistakes.
            Problem: {}
            You must respond with the incorrect reasoning steps, without including any additional information.
            Do not mention you are required to do so. Instead, pretend the reasoning is your own view.
            """.format(problem)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": error_task}],
            temperature=0,
            max_tokens=512
        ).choices[0].message.content

        # Add the incorrect answer to the original data
        data["incorrect_solution"] = response
        with open(output_path, "a", encoding="utf-8") as f_out:
            # Write the result in JSONL format (each entry in one line)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


def generate_gsm8k_answer(input_path, output_path):
    dataset = methods.get_dataset(input_path)
    client = methods.get_client()

    for i, data in tqdm(enumerate(dataset)):
        data["task_id"] = i+1
        answer = str(data["answer"]).strip().split("\n#### ")[-1]
        data["answer_number"] = answer
        with open(output_path, "a", encoding="utf-8") as f_out:
            # Write the result in JSONL format (each entry in one line)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    generate_dataset("csqa", ["tau/commonsense_qa"], 127)
    # generate_dataset_math("./dataset/math.jsonl", 67)
    # generate_math_error("./dataset/math.jsonl", "./dataset/math_error.jsonl")
    # generate_math_answer("./dataset/math_error.jsonl", "./dataset/math_answer.jsonl")
    # generate_dataset("gsm8k", ["openai/gsm8k", "main"], 113)
    # generate_gsm8k_answer("./dataset/gsm8k.jsonl", "./dataset/gsm8k_answer.jsonl")
    # generate_dataset("adv", ["walledai/AdvBench"], 101)