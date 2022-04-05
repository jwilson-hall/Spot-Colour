import os
dir_path = os.path.dirname(os.path.realpath(__file__))

files = os.listdir(dir_path+"\\v5\\")
i = 1

# for file in files:
#     print(file)
print(dir_path)
for file in files:
    if "_p.jpg" in file:
        print(file)
        os.remove(dir_path+"\\v5\\"+file)
        #i = i+1
