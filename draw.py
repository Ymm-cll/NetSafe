import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import evaluate
import methods


def SAA_heatmap(metric, agent_labels=None, round_labels=None):
    """
    Parameters:
    - metric: 2D numpy array,
    - agent_labels: list of str,
    - round_labels: list of str,
    """

    num_agents, num_rounds = metric.shape
    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(num_agents)]
    if round_labels is None:
        round_labels = [f"Iteration {i + 1}" for i in range(num_rounds)]

    data = pd.DataFrame(metric, index=agent_labels, columns=round_labels)
    sns.set_theme(style="whitegrid")


    plt.figure(figsize=(10, 6))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    ax = sns.heatmap(data, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, cbar_kws={'shrink': 0.8})

    plt.title("Agent Performance Across Iterations", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Rounds", fontsize=24, labelpad=15)
    plt.ylabel("Agents", fontsize=24, labelpad=15)

    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=45, fontsize=18)

    cbar = ax.collections[0].colorbar
    cbar.set_label('Performance Metric', size=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()


def SAA_linechart(metrics):
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', 'h', 'H']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'lime', 'black', 'yellow']

    plt.rc('font', family='Comic Sans MS')

    plt.figure(figsize=(8, 8))
    num_agents, num_iterations = metrics.shape


    for i in range(num_agents):
        if i == 0:
            plt.plot(range(1, num_iterations + 1), metrics[i], label=f'Agent {i + 1}', linewidth=3.5,
                     marker=markers[i % len(markers)], markersize=13, color='red', alpha=0.75)
        else:
            plt.plot(range(1, num_iterations + 1), metrics[i], label=f'Agent {i + 1}', linewidth=3.5,
                     marker=markers[i % len(markers)], markersize=13, color=colors[i % len(colors)], alpha=0.75)

    arrow_x = num_iterations // 2
    arrow_y = metrics[0, arrow_x - 1]
    plt.annotate('Attacker Node', xy=(arrow_x - 1, arrow_y + 1), xytext=(arrow_x, arrow_y + 6),
                 arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.1, width=5, headwidth=14), fontsize=24,
                 color='red', fontweight='bold')

    plt.title("SAA over Iterations (Fact/Star)", fontsize=24, fontweight='bold', pad=5)
    plt.xlabel("Iterations", fontsize=28, fontweight='bold', labelpad=5)
    plt.ylabel("Performance (SAA)", fontsize=28, fontweight='bold', labelpad=0)

    plt.xticks(range(1, num_iterations + 1), fontsize=24)
    plt.yticks(range(10, 110, 10), fontsize=24)

    plt.legend(title='Agents', fontsize=20, title_fontsize=20, loc='best', framealpha=0.5)

    plt.grid(visible=True, which='both', color='grey', linestyle='-', linewidth=0.7, alpha=0.7)

    plt.tight_layout()

    plt.show()


def MJA_last_column(data, dataset):
    plt.rc('font', family='Comic Sans MS')

    graph_types = ['chain', 'circle', 'tree', 'star', 'complete']
    graph_display = ["Chain", "Circle", "Binary Tree", "Star Graph", "Complete Graph"]
    if dataset == "gsm8k":
        nums = [0, 1, 2, 3, 4, 5]
    if dataset == "bias":
        nums = [0, 1, 2, 3, 4]

    grouped_data = {graph: [] for graph in graph_types}
    for key, value in data.items():
        if dataset == "gsm8k":
            graph, total, attacker = key.split('_')
            grouped_data[graph].append((int(attacker), value))
        if dataset == "bias":
            graph, total, attacker = key.split('_')
            grouped_data[graph].append((int(total), value))

    for graph in grouped_data:
        grouped_data[graph].sort()
    print(grouped_data)
    bar_width = 0.16
    indices = np.arange(int(len(graph_types))) * 1.2

    colors = ['#ffdd55', '#55aaff', '#cc66ff', '#77cc77', '#ff6666', '#66ccff']
    if dataset == "gsm8k":
        hatch_patterns = ['\\', '++', 'xx', '//', '-', '.']
    if dataset == "bias":
        hatch_patterns = ['-', '.', '//', '\\', 'xx', '++']

    fig, ax = plt.subplots(figsize=(18, 4))

    for i, num in enumerate(nums):
        values = [grouped_data[graph][num][1] for graph in graph_types]
        if dataset == "gsm8k":
            label = num
        if dataset == "bias":
            label = num + 5
        bars = ax.bar(indices + i * bar_width, values, bar_width, color=colors[i % len(colors)],
                      hatch=hatch_patterns[i % len(hatch_patterns)], label=label,
                      edgecolor='black')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10)
            if dataset == "gsm8k":
                ax.text(bar.get_x() + bar.get_width() / 2, -8, i, ha='center', va='bottom', fontsize=12)
            if dataset == "bias":
                ax.text(bar.get_x() + bar.get_width() / 2, -8, i + 5, ha='center', va='bottom', fontsize=12)

    ax.set_xlabel('Attakcer Number/Graph Type', fontsize=16, weight='bold', labelpad=0)
    ax.set_ylabel('Last Iteration MJA', fontsize=16, weight='bold')
    if dataset == "gsm8k":
        ax.set_title('Converged Performance on GSM8k Dataset with Different Attacker Node Numbers (num_total=6)',
                     fontsize=16, weight='bold')
    if dataset == "bias":
        ax.set_title('Converged Performance on Bias Dataset with Different Normal Node Numbers (num_attacker=1)',
                     fontsize=16, weight='bold')
    ax.set_xticks(indices + (len(nums) - 1) * bar_width / 2)
    ax.set_xticklabels(graph_display, fontsize=18, weight='bold')
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7, alpha=0.3)  # 添加灰色网格线
    ax.set_yticklabels([str(i) for i in range(0, 101, 10)], fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14, labelrotation=8, pad=10)
    ax.set_xlim([indices[0] - bar_width, indices[-1] + len(nums) * bar_width])

    if dataset == "gsm8k":
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title="Attacker\nNode\nNumber",
                  title_fontsize=14, ncol=1)
    if dataset == "bias":
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title="Normal\nNode\nNumber",
                  title_fontsize=14, ncol=1)

    plt.tight_layout()
    plt.show()


def SAA_radar(data, title=""):
    plt.rc('font', family='Comic Sans MS')
    num_turns, num_categories = data.shape

    # Angles for the radar chart (one per category)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle by adding the first angle to the end

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for turn in range(num_turns):
        values = data[turn].tolist()
        values += values[:1]  # Close the circle by adding the first value to the end

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Turn {turn + 1}')
        ax.fill(angles, values, alpha=0.25)

    # Add labels to the chart
    display = ['harassment', 'harassment/\nthreatening', 'hate', 'hate/\nthreatening', 'self-harm',
               'self-harm/\ninstructions', 'self-harm/\nintent', 'sexual', 'sexual/\nminors', 'violence',
               'violence/\ngraphic']

    category_labels = display

    # Remove angle tick labels but keep radial lines
    ax.set_xticks(angles[:-1])  # Keep radial lines
    ax.set_xticklabels([])  # Remove angle tick labels

    # Generate a range of red colors from dark to light
    colors = [
        '#FF4500',  # harassment - bright orange
        '#FF6347',  # harassment/threatening - tomato red
        '#8B0000',  # hate - dark red
        '#B22222',  # hate/threatening - firebrick red
        '#9400D3',  # self-harm - dark violet
        '#8A2BE2',  # self-harm/instructions - blue violet
        '#4B0082',  # self-harm/intent - indigo
        '#FFD700',  # sexual - gold
        '#FFA500',  # sexual/minors - orange
        '#DC143C',  # violence - crimson
        '#8B008B'  # violence/graphic - dark magenta
    ]

    # Manually add text labels at each angle with different shades of red
    for angle, label, color in zip(angles[:-1], category_labels, colors):
        ax.text(angle, ax.get_ylim()[1] * 0.9, label, fontsize=20, ha='center', va='center',
                color=color)  # Adjust position inward

    # Add a legend and title with increased font size
    ax.legend(loc='upper left', fontsize=16, framealpha=0.5, ncol=2)
    # Increase legend font size
    ax.set_title('Moderation Score over Iterations\n' + title, size=20, color='black', pad=10,
                 weight="bold")  # Increase title font size

    # Increase the radial and angular tick labels font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.text(0.5, 0.08, 'Note: Data values are scaled by 10', ha='center', fontsize=14, color='black')

    plt.show()


if __name__ == "__main__":
    evaluation = "dynamic_SAA"
    draw_type = "radar"
    # evaluation = "static_PV"
    sample_ids = [3]
    dataset = "adv"
    model = "gpt-4o-mini"
    if dataset == "adv":
        model = "gpt-3.5-turbo"
    graph_types = ["chain", "circle", "tree", "star", "complete"][-1:]
    agent_nums = [6]
    agent_num = 6
    attacker_nums = [5]
    attacker_num = 5
    dataset_path = f"./dataset/{dataset}.jsonl"
    if "SAA" in evaluation:
        for graph_type in graph_types:
            print(f"Graph: {graph_type}_{agent_num}, Attacker Number: {attacker_num}")
            if "static" in evaluation:
                type = evaluation.split("_")[-1]
                metric = evaluate.static_evaluate(methods.generate_adj(agent_num, graph_type),
                                                  list(range(attacker_num + 1)), type)
                print(f"Metric ({type}):", metric)
            else:
                type = evaluation.split("_")[-1]
                metrics = []
                for sample_id in sample_ids:
                    if dataset == "adv":
                        output_path = f"moderation/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                    else:
                        output_path = f"output/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                    accuracy = evaluate.evaluate(dataset_path, output_path, attacker_num, type)
                    metrics.append(accuracy)
                metrics = np.array(metrics)
                mean = np.mean(metrics, axis=0)
                if dataset != "adv":
                    mean = np.round(100 * mean, 2)
                else:
                    mean = mean * 10
                print("Mean", mean)
                if draw_type == "linechart":
                    SAA_linechart(mean)
                if draw_type == "radar":
                    SAA_radar(np.mean(mean[:5], axis=0), "(AdvBench/Attacker Nodes Average)")
                    SAA_radar(mean[5], "(AdvBench/Singe Normal Node)")
    if "MJA" in evaluation and draw_type == "last_column":
        last_mja = {}
        for agent_num in agent_nums:
            for attacker_num in attacker_nums:
                for graph_type in graph_types:
                    type = evaluation.split("_")[-1]
                    print(f"Graph: {graph_type}_{agent_num}, Attacker Number: {attacker_num}")
                    metrics = []
                    for sample_id in sample_ids:
                        if dataset == "adv":
                            output_path = f"moderation/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                        else:
                            output_path = f"output/{model}/{dataset}/{sample_id}/{dataset}_{graph_type}_{agent_num}_{attacker_num}.output"
                        accuracy = evaluate.evaluate(dataset_path, output_path, attacker_num, type)
                        metrics.append(accuracy)
                    metrics = np.array(metrics)
                    mean = np.mean(metrics, axis=0)
                    mean = np.round(100 * mean, 2)
                    last_mja[f"{graph_type}_{agent_num}_{attacker_num}"] = mean[-1]
        print(last_mja)
        MJA_last_column(last_mja, dataset)
