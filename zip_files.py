import os
files = os.listdir()
for i, f in enumerate(files):
    print("zip -r {i}.zip {f}".format(i=i, f=f))