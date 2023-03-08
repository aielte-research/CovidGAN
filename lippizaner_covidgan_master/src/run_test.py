import subprocess
import os 
import sys
import datetime 
import hashlib
import time

now = datetime.datetime.now()
#print("NEPTUNE_CUSTOM_RUN_ID = ", os.environ['NEPTUNE_CUSTOM_RUN_ID'])
hash =  hashlib.md5(now.strftime("%Y-%m-/%d_%H_%M_%S").encode('utf_8')).hexdigest()
os.environ['NEPTUNE_CUSTOM_RUN_ID'] = hash
print("NEPTUNE_CUSTOM_RUN_ID = ", os.environ['NEPTUNE_CUSTOM_RUN_ID'])

number_of_clients = int(sys.argv[1])
file_route = sys.argv[2]

lippi_clients = []

for i in range(number_of_clients):
    temp_client = subprocess.Popen(["python", "main.py", "train", "--distributed", "--client"], text=True)
    lippi_clients.append(temp_client)
print("Client(s) started")
lippi_master = subprocess.Popen(["python", "main.py",  "train", "--distributed", "--master","-f", file_route], text=True)
print("master_started")

while(lippi_master.poll() is None):
    time.sleep(2)

for client in lippi_clients:
    client.kill()
    print("Client: ", client)

print(lippi_master) 
lippi_master.kill()



#subprocess.run("conda deactivate", shell =True)
#print("deactivated")
#subprocess.run("conda activate find_out", shell = True)
#print("activated")

#TODO Now run lipizzaner ??
#subprocess.run(f"python main.py train --distributed --client; python main.py train --distributed --master -f configuration/covid-qu-debug/covidqu_1.yml;", shell=True)


#subprocess.run(f"anaconda-project set-variable NEPTUNE_CUSTOM_RUN_ID = {hash}", shell=True)
#print("NEPTUNE_CUSTOM_RUN_ID = ", os.environ['NEPTUNE_CUSTOM_RUN_ID'])