from utils.gpt4_util import *
import time
import requests
import json
import random


class GPT4:
    def __init__(self, user="XXX", max_time=100, debug=False, version="gpt-4o"):
        self.user = user
        self.max_time = max_time
        self.debug = debug
        self.version = version

    def __call__(self, content, system_info, infos="Test Query"):
        # build query
        content = content.replace("\\","\\\\")
        msg = "[{\"role\":\"system\",\"content\":\"" + system_info + "\"},{\"role\":\"user\",\"content\":\"" + content + "\"}]"
        str_id = self.user + str(int(time.time())) + str(random.randrange(100000, 1000000))

        # send query
        headers = {'Content-Type': 'application/json'}
        param = build_req_param(str_id, msg, self.version,"online")
        post_data = {"encryptedParam": aes_encrypt(json.dumps(param))}
        req_response = requests.post('XXXXXXXXXXXXXXXXXXXXXXXX', data=json.dumps(post_data), headers=headers)

        if "MESSAGE_FORMAT_ERROR" in str(req_response.content):
            print("[++++]req response error: MESSAGE_FORMAT_ERROR")
            print(f"[DEBUG INFO] req_response.content: {req_response.content}")
            return None
        if self.debug:
            print(f"[DEBUG INFO] req_response.content: {req_response.content}")

        # get result
        pull_param = build_pull_param(str_id, "online")
        request_times = 0
        while request_times<=self.max_time:
            post_data = {"encryptedParam": aes_encrypt(json.dumps(pull_param))}
            pull_response = requests.post('XXXXXXXXXXXXXXXXXXXXXXXX',
                                            data=json.dumps(post_data),
                                            headers=headers)

            resp_content = json.loads(pull_response.content)
            if 'data' not in resp_content:
                print("[-------]pull response error \n msg id: {}".format(str_id))
                continue

            if 'response' in resp_content['data']['values']:
                print("Got pull response!")
                for key in resp_content['data']['values']:
                    if key == "response":
                        try:
                            rtn= resp_content['data']['values'][key]
                        except:
                            print("[++++]pull response error \n msg id: {}".format(str_id))
                break
            if infos:
                print(f"[{infos}, Time {request_times}/{self.max_time}] Wait for pull response... ")
            else:
                print(f"[Time {request_times}/{self.max_time}] Wait for pull response... ")
            request_times +=1
            time.sleep(1)
        try:
            rtn = rtn.replace("&quot;", "\"").split("\"message\":{\"role\":\"assistant\",\"content\":\"")[1].split("\"}}],\"system_fingerprint")[0]
            rtn = rtn.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
        except:
            rtn = None
        return rtn
    
    def debug_on(self):
        self.debug = True
    
    def debug_off(self):
        self.debug = False

