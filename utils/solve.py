import os
import subprocess
import time
import random


class PyomoSolver:
    def __init__(self):
        self.solver_id = str(int(time.time()))

    def __call__(self, code):
        if not os.path.exists(f"./buffer/"):
            os.makedirs(f"./buffer/", exist_ok=True)
        
        path = f"./buffer/test_{self.solver_id}_{random.randrange(100000, 1000000)}.py"

        with open(path, "w") as f:
            f.write(code)
        ans = subprocess.run(f"python {path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(path)

        return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))
