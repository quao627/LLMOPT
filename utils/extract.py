class Extractor:
    def __init__(self):
        pass

    def __call__(self, msg):
        try:
            if "```python" in msg:
                return msg.split("```python")[1].split("```")[0]
            elif "```plaintext" in msg:
                return msg.split("```plaintext")[1].split("```")[0]
            elif "```text" in msg:
                return msg.split("```text")[1].split("```")[0]
            elif "```" in msg:
                return msg.split("```")[1].split("```")[0]
            else:
                return None
        except:
            return None
                
    def extract(self, msg):
        if "```" in msg:
            return msg.split("```")[1].split("```")[0]
        else:
            return None

    def extract_text(self, msg):
        if "```text" in msg:
            return msg.split("```text")[1].split("```")[0]
        else:
            return self.extract(msg)

    def extract_plain_text(self, msg):
        if "```plaintext" in msg:
            return msg.split("```plaintext")[1].split("```")[0]
        else:
            return self.extract_text(msg)

    def extract_python(self, msg):
        if "```python" in msg:
            return msg.split("```python")[1].split("```")[0]
        else:
            return self.extract(msg)