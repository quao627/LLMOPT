from utils.gpt4_infer import GPT4
from utils.template import ClassificationTypeTemplate, ClassificationBackgroundTemplate
import json
import os
from collections import Counter


templater_type = ClassificationTypeTemplate()
templater_bk = ClassificationBackgroundTemplate()
gpt4 = GPT4()


dir_path = "./dir_path"


datasets = [
    'mamo_complexlp',
    'nl4opt_test',
    'complexor',
    'industryor_test',
    'mamo_easylp',
    'nlp4lp',
]

for dataset in datasets:
    path = f"{dir_path}/{dataset}.json"

    bks = []
    tys = []
    s = 0
    
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            bks.append(data['background'])
            tys.append(data['type'])
            s += 1
    
    print("+"*50)
    print(dataset)
    print(dict(Counter(bks)))
            

for dataset in datasets:
    path = f"{dir_path}/{dataset}.json"

    bks = []
    tys = []
    s = 0
    
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            bks.append(data['background'])
            tys.append(data['type'])
            s += 1
    
    print("+"*50)
    print(dataset)
    print(dict(Counter(tys)))

