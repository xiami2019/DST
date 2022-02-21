import json

domains = {'餐馆': 'restaurant', '景点': 'attraction', '酒店': 'hotel', '地铁': 'metro', '出租': 'taxi'}

ontology = {}

for domain in domains.keys():
    ontology[domain] = {}

# extract slot value from db
for domain, domain_file_name in domains.items():
    file_name = domain_file_name + '_db.json'

    db = json.loads(open(file_name, encoding='utf-8').read())

    for record in db:
        '''
        record[0] = the entry's names
        record[1] = a list of slot value pair 
        '''
        for slot_name, slot_value in record[1].items():
            if slot_name == '领域':
                continue

            if slot_name not in ontology[domain]:
                ontology[domain][slot_name] = set()
            
            if isinstance(slot_value, list):
                for value in slot_value:
                    ontology[domain][slot_name].add(value)
            else:
                ontology[domain][slot_name].add(slot_value)

# extract slot value from source data
train_data = json.loads(open('./../train.json', encoding='utf-8').read())
val_data = json.loads(open('./../val.json', encoding='utf-8').read())
test_data = json.loads(open('./../test.json', encoding='utf-8').read())

all_data = [train_data, val_data, test_data]

for dialog_data in all_data: # train、val、test；
    for id, session in dialog_data.items():
        for turn in session['messages']:
            for slot_info in turn['dialog_act']:
                intent, domain, slot_name, slot_value = slot_info
                if '-' in slot_name:
                    slot_name, slot_value = slot_name.split('-')

                if slot_value != "" and slot_value != 'none':
                    if slot_name not in ontology[domain]:
                        ontology[domain][slot_name] = set()

                    ontology[domain][slot_name].add(slot_value)

for domain in ontology.keys():
    for slot_name in ontology[domain].keys():
        ontology[domain][slot_name] = list(ontology[domain][slot_name])

with open('ontology.json', 'w') as f:
    json.dump(ontology, f, ensure_ascii=False, indent=2)

print("Creating ontology Over!")

        