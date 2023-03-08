import os
 

for x in os.environ:
    print((x, os.getenv(x)))