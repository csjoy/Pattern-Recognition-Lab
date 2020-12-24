import os
import sys
import random

file_list = os.listdir()
os.mkdir('Formatted')

for files in file_list:
    file_info = files.split('.')
    if file_info[1] == 'txt':
        file_type = file_info[0].split('_')

        try:
            fh = open(f'{file_info[0]}.txt', 'r')
            fw = open(f'Formatted/{file_info[0]}_formatted.txt', 'w')
        except Exception:
            print("Couldn't open file.")
            quit()

        while True:
            line = fh.readline()
            if line == "":
                break
            
            if line[0]==">":
                line = line[:-1]+"|"+str(random.randint(0,1))+"|"+file_type[1]+"ing"+"\n"
            
            fw.write(line)
        
        fh.close()
        fw.close()