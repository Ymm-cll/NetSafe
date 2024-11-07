# NetSafe
├── dataset						: direcotry for experiment datasets
├── README.md						: introduction
├── calculate_sim.py						: code for calculating kendall's tau
├── cut_figure.py						: code for cutting figures for writing
├── draw.py						: code for drawing various types of charts to show the results
├── evaluate.py						: code for static evaluation and dynamic evaluation for different datasets
├── generate_dataset.py						: code for dataset generation/sampling
├── methods.py						: code for some general functions
├── moderation.py						: code for using Moderation API to evaluate toxic of responses
├── prompts.py						: system prompts
├── run.py						: code for runing with one-shot
├── run_adv.py						: code for runing experiments on advbench dataset
├── run_bias.py						: code for runing experiments on bias dataset
├── run_csqa.py						: code for runing experiments on csqa dataset
├── run_fact.py						: code for runing experiments on fact dataset
└── run_gsm8k.py						: code for runing experiments on gsm8k dataset

# Quick Start
1. Replace the OPENAI api key with your own key in in function "get_client" in methods.py
2. Set parameters in run.py
3. Run run.py 
