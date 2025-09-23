from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import sys
import os
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prompts import generate_prompt
from openai import OpenAI
from llm import create_llm_client

llm_client = create_llm_client("openai")


client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")



# load model and tokenizer
path = '/orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/LLMOPT-Qwen2.5-14B'
path_t = '/orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/LLMOPT-Qwen2.5-14B'
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(path_t)


# inference to get five elements
def infer_five_elem(question):
    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question)}
    ]
    resp = client.chat.completions.create(model="LLMOPT-Qwen2.5-14B",messages=messages,max_tokens=8192,)

    response = resp.choices[0].message.content

    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


# inference to get pyomo python code
def infer_code(five_elem):
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem)}
    ]
    resp = client.chat.completions.create(model="LLMOPT-Qwen2.5-14B",messages=messages,max_tokens=8192,)
    response = resp.choices[0].message.content

    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')


# execute the code
def test_code(code_str):
    code_path = f"./test.py"
    with open(code_path, "w") as f1:
        f1.write(code_str)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))


# example usage
def eval_single_problem(problem):
    question = problem['description']
    answer = problem['ground_truth']
    five_elem = infer_five_elem(question)
    code_str = infer_code(five_elem)
    out_log, err_log = test_code(code_str)
    if err_log != "":
        return 'error'
    check_answer_prompt = f"check the answer contained in the following logs is correct or not. the ground truth answer is {answer}. the logs is {out_log}. output the result in True/False. write in json: {{'correct': 'True/False'}}. Output:"
    check_answer_response = llm_client.generate(check_answer_prompt, force_json=True)
    check_answer_response = json.loads(check_answer_response)
    if check_answer_response['correct']:
        return 'correct'
    else:
        return 'wrong'


def eval(data_dir, dataset_file, max_workers=None):
    dataset_path = os.path.join(data_dir, dataset_file)
    with open(dataset_path, "r") as f:
        problems = json.load(f)
    problems = problems

    total_num = len(problems)
    correct_num = 0
    wrong_answer_num = 0
    runtime_error_num = 0
    
    # Thread-safe lock for result collection
    result_lock = Lock()
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all problems to the thread pool
        future_to_problem = {
            executor.submit(eval_single_problem, problem): problem 
            for problem in problems
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_problem):
            result = future.result()
            with result_lock:
                if result == "correct":
                    correct_num += 1
                elif result == "wrong":
                    wrong_answer_num += 1
                elif result == "error":
                    runtime_error_num += 1
    
    results = {
        "total_num": total_num,
        "correct_num": correct_num,
        "wrong_answer_num": wrong_answer_num,
        "runtime_error_num": runtime_error_num,
        "accuracy": correct_num / total_num,
        "correct_rate_ignore_runtime_error": correct_num / (total_num - runtime_error_num) if (total_num - runtime_error_num) > 0 else 0
    }
    return results


if __name__ == "__main__":
    # Create a command line parameter parser
    parser = argparse.ArgumentParser(description="Run the optimization problem")
    parser.add_argument("--dir", type=str, help="Directory of the problem")
    parser.add_argument("--data_dir", type=str, default="../baseline_test_data", help="Directory containing baseline test data")
    parser.add_argument("--dataset_file", type=str, default="nlp4lp_test.json", help="JSON file with test problems")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of concurrent threads (default: None for auto)")
    args = parser.parse_args()
    results = eval(args.data_dir, args.dataset_file, args.max_workers)
    os.makedirs(f"../results/llmopt/{args.dataset_file.replace('.json', '')}", exist_ok=True)
    with open(f"../results/llmopt/{args.dataset_file.replace('.json', '')}/{args.dataset_file.replace('.json', '_results.json')}", 'w') as f:
        json.dump(results, f, indent=2)