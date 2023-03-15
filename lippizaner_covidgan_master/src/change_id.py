import subprocess
import os 
import hashlib
from datetime import datetime

print("id before change: ", os.environ["NEPTUNE_CUSTOM_RUN_ID"])
now = datetime.now()
hash_id = hashlib.md5(now.strftime("%Y-%m-%d_%H_%M_%S").encode("utf-8")).hexdigest()

with open("neptune_id.txt", "w") as f:
  print(type(hash_id))
  f.write(hash_id)

with open("neptune_id.txt", "r") as f:
  n_id = f.read()
  print(type(n_id))
#cmd_to_run = f"export NEPTUNE_CUSTOM_RUN_ID={hash_id}"
#print(cmd_to_run)
#os.environ["NEPTUNE_CUSTOM_RUN_ID"] = hash_id

#os.system(cmd_to_run)
#subprocess.run("conda deactivate", shell=True)
#subprocess.run(cmd_to_run,shell=True)
#subprocess.run("conda activate neptunelippi", shell=True)
#print("id after change: ", os.environ["NEPTUNE_CUSTOM_RUN_ID"])
