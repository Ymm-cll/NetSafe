import json
import os
import numpy as np
from openai import OpenAI

def get_client(openai_api_key=os.environ['OPENAI_API_KEY']):
    client = OpenAI(api_key=openai_api_key)
    return client


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('')


def generate_adj(n, graph_type):
    if "complete" in graph_type:
        adj_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
    if "tree" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            # Add edges if left and right children are within bounds
            if left_child < n:
                adj_matrix[i][left_child] = 1
                adj_matrix[left_child][i] = 1
            if right_child < n:
                adj_matrix[i][right_child] = 1
                adj_matrix[right_child][i] = 1
    if "chain" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        # Set the values for a chain structure
        for i in range(n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
    if "star" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj_matrix[0][i] = 1
            adj_matrix[i][0] = 1
        for i in range(1, n - 1):
            adj_matrix[i][i + 1] = 1
            adj_matrix[i + 1][i] = 1
        adj_matrix[1][n - 1] = 1
        adj_matrix[n - 1][1] = 1
    if "circle" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj_matrix[i][(i + 1) % n] = 1
            adj_matrix[(i + 1) % n][i] = 1
    return adj_matrix



def get_dataset(ds_path):
    dataset = []
    with open(ds_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            item = eval(line.strip())
            dataset.append(item)
    return dataset

