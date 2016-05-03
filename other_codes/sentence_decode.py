import chardet
import codecs
import os

path = '/home/bero/Desktop/CafeBazaar_comments/'
destpath = '/home/bero/Desktop/bazzardataset'



destfile = open(destpath, "w")

for sourcefile in os.listdir(path):
    filepath = path + sourcefile
    #myfile = codecs.open(filepath, "r", encoding = "windows-1256")
    each_sourcefile = open(filepath, "r")
    outputfile = each_sourcefile.readlines()
    myindex = 0
    for line in outputfile:
        line = line.split('\n')
        line = line[0]
        if(line == "<><>-----<><>"):
            continue
        else:
            if(len(line) > 50):
                print("myindex is: ", myindex)
                myindex = myindex + 1
                print(len(line))
            #line=line.decode(encoding='windows-1256')
            #line=line.encode('UTF-8')
                destfile.write(line)
                destfile.write("\n")
    each_sourcefile.close()

destfile.close()





'''
sourceP_path = '/home/bero/Desktop/mydatasetP'
sourceN_path = '/home/bero/Desktop/mydatasetN'
dest_path = '/home/bero/Desktop/totaldata'
destfile = open(dest_path, 'w')
sourceP_file = open(sourceP_path, 'r')
sourceN_file = open(sourceN_path, 'r')
labelpath = '/home/bero/Desktop/totallabel'
labelfile = open(labelpath, "w")

file_lines = sourceP_file.readlines()
for line in file_lines:
    labelfile.write("1\n")
    destfile.write(line)

file_lines = sourceN_file.readlines()
for line in file_lines:
    labelfile.write("0\n")
    destfile.write(line)

labelfile.close()
destfile.close()

'''
