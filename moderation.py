import json
import threading
from tqdm import tqdm
import methods

lock = threading.Lock()


def process_item(item, client):
    agent_score = {}
    task_id = list(item.values())[0]
    agent_score["task_id"] = task_id
    for agent_key in list(item.keys())[1:]:
        dialogue_temp = item[agent_key]
        dialogue_score = []
        for msg in dialogue_temp:
            if msg["role"] == "assistant":
                score = client.moderations.create(
                    model="text-moderation-latest",
                    input=msg["content"]["response"],
                ).to_dict()
                dialogue_score.append(score)
        agent_score[agent_key] = dialogue_score
    return agent_score


def process_task(item, client, output_path, format):
    result = process_item(item, client)
    with lock:
        with open(output_path, 'a', encoding="utf-8") as f:
            if format:
                f.write(json.dumps(result, indent=4, ensure_ascii=False) + "\n")
            else:
                f.write(str(result) + "\n")


def evaluate_adv(input_path, output_path, format, p):
    client = methods.get_client()
    input_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        temp = f.readlines()
        for item in temp:
            input_data.append(eval(item.strip()))

    threads = []
    for item in tqdm(input_data):
        thread = threading.Thread(target=process_task, args=(item, client, output_path, format))
        threads.append(thread)
        if len(threads) >= p:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.start()
    for t in threads:
        t.join()



if __name__ == "__main__":
    sample_ids = [3]
    dataset = "adv"
    format = False
    model = "gpt-3.5-turbo"
    graph_types = ["chain", "circle", "tree", "star", "complete"][-1:]
    agent_num = 6
    p = 8
    attacker_num = agent_num - 1
    for graph_type in graph_types:
        for sample_id in sample_ids:
            input_path = f"output/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
            output_path = f"moderation/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
            methods.create_directory(f"moderation/{model}/{dataset}/{sample_id}")
            evaluate_adv(input_path, output_path, format, p)
