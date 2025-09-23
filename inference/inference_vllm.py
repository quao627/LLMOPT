from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prompts import generate_prompt
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

print(resp.choices[0].message.content)



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

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # generated_ids = model.generate(
    #     model_inputs.input_ids,
    #     max_new_tokens=8192
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # generated_ids = model.generate(
    #     model_inputs.input_ids,
    #     max_new_tokens=8192
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
question = "Consider a problem where we have a set `P`. For each element `j` in `P`, we have a parameter `a[j]`, a parameter `c[j]`, and a parameter `u[j]`. We also have a global parameter `b`. We have a variable `X[j]` for each `j` in `P`. The goal is to maximize the total profit, which is the sum of `c[j] * X[j]` for all `j` in `P`. The constraints are that the sum of `(1/a[j]) * X[j]` for all `j` in `P` should be less than or equal to `b`, and `X[j]` should be between 0 and `u[j]` for all `j` in `P`.\n\nThe following parameters are included in this problem:\na: a list of integers, parameter for each element in set P\nc: a list of integers, profit coefficient for each element in set P\nu: a list of integers, upper limit for each element in set P\nb: an integer, the global constraint parameter\n\n\nThe following data is included in this problem:\n{'a': [3, 1, 2], 'c': [5, 10, 8], 'u': [4, 6, 3], 'b': 4}"
five_elem = infer_five_elem(question)
code_str = infer_code(five_elem)
out_log, err_log = test_code(code_str)
print(out_log)
