import os

cpptest="pycall.exe" #in linux without suffix .exe

if os.path.exists(cpptest):
    f=os.popen(cpptest)
    data=f.readlines() #read the C++ printf or cout content
    f.close()
    print(data)

print("python execute cpp program:")
os.system(cpptest)