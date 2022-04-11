import os
dir_path = os.path.dirname(os.path.realpath(__file__))

files = os.listdir(dir_path+"\\")
i = 12

# for file in files:
#     print(file)
print(dir_path)
j = 1
for file in files:
    if ".jpg" in file:
        
        
        newName = dir_path+"\\"+str(i)+"_gt"+str(j)+".jpg"
        os.rename(dir_path+"\\"+file,newName)
        
        j += 1
        if j == 6:
            j = 1
            i += 1
