from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)



DEV_VISIT = {
    "visitDomain": "XXXXXX",
    "visitBiz": "XXXXXX",
    "visitBizLine": "XXXXXX"
}
ONLINE_VISIT = {
    "visitDomain": "XXXXXX",
    "visitBiz": "XXXXXX",
    "visitBizLine": "XXXXXX"
}
API_KEY = "XXXXXX"



def aes_encrypt(data, key="XXXXXX"):
    iv = "XXXXXX"
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    block_size = AES.block_size

    if len(data) % block_size != 0:
        add = block_size - (len(data) % block_size)
    else:
        add = 0
    data = data.encode('utf-8') + b'\0' * add
    encrypted = cipher.encrypt(data)
    result = b2a_hex(encrypted)
    return result.decode('utf-8')


def aes_decode(data, key="XXXXXX"):
    iv = 'XXXXXX'
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    result2 = a2b_hex(data)
    decrypted = cipher.decrypt(result2)
    return decrypted.rstrip(b'\0')


def build_req_param(str_id, messages, model="gpt-4-turbo", mode="dev"):
    param = {
        "serviceName": "asyn_chatgpt_prompts_completions_query_dataview",
        "cacheInterval": "-1",
        "queryConditions": {
            "model": model,
            "messages": messages,
            "max_tokens": "4096",
            "temperature": "1",
            "n": "1",
            "top_p": "1",
            "outputType": "PULL",
            "messageKey": str_id,
            "api_key": API_KEY,
        }
    }
    param.update(DEV_VISIT if mode != "online" else ONLINE_VISIT)
    return param


def build_pull_param(str_id, mode="dev"):
    param = {
        "serviceName": "chatgpt_response_query_dataview",
        "cacheInterval": "-1",
        "queryConditions": {
            "messageKey": str_id
        }
    }
    param.update(DEV_VISIT if mode != "online" else ONLINE_VISIT)
    return param

