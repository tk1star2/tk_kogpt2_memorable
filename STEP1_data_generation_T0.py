import openpyxl
import random
from openpyxl import Workbook, load_workbook
import glob
import json


def D3_wellness_dialog_for_autoregressive():
  folder_path = "./TK_data/TT_data"
  output_path = "./TK_data/T0_data/T0_data.txt"

  output_file = open(output_path, 'w')
  output_file.close()
  output_file = open(output_path, 'a', encoding='utf-8')



  for file_path in glob.glob(folder_path + "/*.txt"):
    print("\n\n\n {} \n\n\n".format(file_path))
    file = open(file_path, 'r', encoding='utf-8')
    ques_lines = file.readline()
    while True:
        answ_lines = file.readline()
        if not answ_lines:
            break
        output_file.write(ques_lines[:-1] + "    " + answ_lines[:-1] + "\n")
        ques_lines = answ_lines 
   
    file.close()
    output_file.write("<CONTEXT_END>\n")

  output_file.close()



if __name__ == "__main__":
  D3_wellness_dialog_for_autoregressive()
