import json 
import glob

TARGET ="./TK_data/T3_national"

for json_path in glob.glob(TARGET+"/JSON/*.json"):
    print(json_path)
    txt_path = TARGET +'/TXT/' +json_path.split('/')[-1].split('.')[-2] +'.txt'

    with open(json_path, 'r') as json_file :
        py_data = json.load(json_file)
        py_document = py_data['document'][0]
        py_utterance = py_document['utterance']
        with open(txt_path, 'w') as txt_file :
            for utterance in py_utterance :
                txt_file.write(utterance['form']+'\n')
            
    
