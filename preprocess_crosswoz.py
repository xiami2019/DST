import os
import json
from tqdm import tqdm
from collections import defaultdict


def get_turn_domains(message):
    domains = set()
    for act in message['dialog_act']:
        domain = act[1]
        if domain != 'greet' and domain != 'thank' and domain != 'welcome' and domain != 'bye' and domain != 'reqmore':
            domains.add(domain)
    return list(domains)

class PreprocessorDST(object):
    def __init__(self) -> None:
        self.data_dir = './crosswoz/'
        self.save_data_dir = os.path.join(self.data_dir, 'processed')
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)
    
    def process(self):
        with open("./crosswoz/train.json") as f:
            train_data = json.load(f)
        with open("./crosswoz/val.json") as f:
            val_data = json.load(f)
        with open("./crosswoz/test.json") as f:
            test_data = json.load(f)

        self.extract_belief_states_labels()

        train_data = self.convert_data_to_MultiWOZ(train_data, self.save_data_dir, 'train')
        val_data = self.convert_data_to_MultiWOZ(val_data, self.save_data_dir, 'val')
        test_data = self.convert_data_to_MultiWOZ(test_data, self.save_data_dir, 'test')

        train_data = self.generate_dst_data(train_data, 'train')
        val_data = self.generate_dst_data(val_data, 'val')
        test_data = self.generate_dst_data(test_data, 'test')

    def extract_belief_states_labels(self):
        self.all_states_labels = {}

        with open("./crosswoz_dst/train_dials.json") as f:
            train_data = json.load(f)
        with open("./crosswoz_dst/dev_dials.json") as f:
            val_data = json.load(f)
        with open("./crosswoz_dst/test_dials.json") as f:
            test_data = json.load(f)

        all_data = {'train': train_data, 'val': val_data, 'test': test_data}
        for data_type in ['train', 'val', 'test']:
            data = all_data[data_type]
            for dialog in data:
                self.all_states_labels[dialog['dialogue_idx']] = []
                for round in dialog['dialogue']:
                    self.all_states_labels[dialog['dialogue_idx']].append(round['belief_state'])

    def check_if_inform_slots(self, slot_domain, slot_name, all_inform_slots):
        if slot_domain not in all_inform_slots:
            return False
        elif slot_name not in all_inform_slots[slot_domain]:
            return False
        else:
            return True

    def remove_space(self, text):
        new_text = ''
        for i in text:
            if i != ' ':
                new_text += i
        return new_text

    def convert_act_to_span(self, acts):
        '''
        convert dialog acts or system acts to span
        '''
        result_span = ''
        act_dict = defaultdict(dict)
        for act in acts:
            token1, token2 = act.split('-')
            token1, token2 = token1.lower(), token2.lower()
            if token2 == 'general':
                act_dict[token2][token1] = {}
            else:
                if token2 not in act_dict[token1]:
                    act_dict[token1][token2] = []
                for sub_act in acts[act]:
                    act_dict[token1][token2].append(sub_act[0])

        for domain in act_dict:
            result_span += '[' + domain + '] '
            for intent in act_dict[domain]:
                result_span += '[' + intent + '] '
                
                if domain != 'general' and intent != 'nooffer':
                    for slot_name in act_dict[domain][intent]:
                        result_span += slot_name + ' '

        return result_span

    def convert_belief_states_to_span(self, belief_states):
        belief_states_span = ''
        belief_states_dict = defaultdict(list)
        for slot in belief_states:
            slot_pair = slot['slots'][0]
            slot_domain_name, slot_value = slot_pair
            domain, slot_name = slot_domain_name.split('-')
            belief_states_dict[domain].append((slot_name, slot_value))

        for domain in belief_states_dict:
            belief_states_span += '[' + domain + '] '
            for slot_pair in belief_states_dict[domain]:
                slot_name, slot_value = slot_pair
                slot_value = self.remove_space(slot_value)
                belief_states_span += '[' + slot_name + '] ' + slot_value + ' '
        
        return belief_states_span

    def generate_dst_data(self, data, data_type):
        print("Generating {:s} data of MTTOD format for DST".format(data_type))
        save_path = os.path.join(self.save_data_dir, data_type + '_mttod.json')
        dst_data = {}
        for dialogue_id, session in tqdm(data.items()):
            dst_data[dialogue_id] = {}

            # add goal and extract inform slots
            inform_slots = {}
            dst_data[dialogue_id]['goal'] = {}
            for sub_goal in session['goal']:
                if session['goal'][sub_goal] != []:
                    dst_data[dialogue_id]['goal'][sub_goal] = session['goal'][sub_goal]
                    inform_slots[sub_goal] = set()
                    for item in session['goal'][sub_goal]:
                        if 'inform' in item:
                            for slot_name in item['inform']:
                                inform_slots[sub_goal].add(slot_name)

            # add log
            log = []
            single_turn = {}
            for i, round in enumerate(session['log']):
                if i % 2 == 0:
                    # usr
                    turn_num = len(log)
                    single_turn['turn_num'] = turn_num
                    single_turn['user'] = round['text']
                    single_turn['user_act'] = self.convert_act_to_span(round['dialog_act'])
                    single_turn['belief_state'] = self.convert_belief_states_to_span(self.all_states_labels[dialogue_id][turn_num])
                else:
                    # sys
                    single_turn['resp'] = round['text']
                    single_turn['sys_act'] = self.convert_act_to_span(round['dialog_act'])
                    log.append(single_turn.copy())
                    single_turn = {}
            dst_data[dialogue_id]['log'] = log

        with open(save_path, 'w') as fp:
            json.dump(dst_data, fp, ensure_ascii=False, indent=4)


    def get_user_goal(self, goal):
        user_goal = {'餐馆': [], '景点': [], '酒店': [], '出租': [], '地铁': []} # A single domain may have two sub goals
        
        for sub_goal in goal:
            sub_goal_id, goal_domain, goal_slot_name, goal_slot_values, is_finished = sub_goal

            if is_finished: # skip finished sub goals
                continue

            if len(user_goal[goal_domain]) == 0 or sub_goal_id != user_goal[goal_domain][-1]['sub_goal_id']:
                user_goal[goal_domain].append({'sub_goal_id': sub_goal_id})
            
            if goal_slot_values == [] or goal_slot_values == '':
                # request slot
                if 'reqt' not in user_goal[goal_domain][-1]:
                    user_goal[goal_domain][-1]['reqt'] = []
                user_goal[goal_domain][-1]['reqt'].append(goal_slot_name)
            else:
                # inform slot
                if 'inform' not in user_goal[goal_domain][-1]:
                    user_goal[goal_domain][-1]['inform'] = {}
                user_goal[goal_domain][-1]['inform'][goal_slot_name] = goal_slot_values

        return user_goal 

    def convert_data_to_MultiWOZ(self, data, processed_dir, data_type):
        '''
        Convert CrossWOZ to the format of MultiWOZ 
        data_type: train / test / val
        return True if successed
        '''

        saved_dir = os.path.join(processed_dir, data_type + '_mwoz.json')

        processed_data = {}
        print("Converting {:s} data to MultiWOZ's format".format(data_type))
        for dialogue_id, dialogue_info in tqdm(data.items()):
            single_session = {}
            # user_goal = {'餐馆': [], '景点': [], '酒店': [], '出租': [], '地铁': []} # A single domain may have two sub goals
            user_goal = self.get_user_goal(dialogue_info['goal'])
            # add user's goal information
            single_session['goal'] = user_goal
            
            log = []
            # add dialogue's turns
            for single_message in dialogue_info['messages']:
                utterance = {}
                # add text
                utterance['text'] = single_message['content']

                # add dialog action
                dialog_actions = {}
                for dialog_act in single_message['dialog_act']:
                    intent, domain, slot_name, slot_value = dialog_act
                    if slot_value == '':
                        # 处理request slot
                        slot_value = '?'
                    action_name = domain + '-' + intent
                    if action_name not in dialog_actions:
                        dialog_actions[action_name] = []
                    dialog_actions[action_name].append([slot_name, slot_value])
                utterance['dialog_act'] = dialog_actions
                
                # add metadata information
                if 'user_state' in single_message:
                    # metadata is user states
                    metadata = self.get_user_goal(single_message['user_state'])
                elif 'sys_state' in single_message:
                    # metadata is system states
                    metadata = single_message['sys_state']
                utterance['metadata'] = metadata
                log.append(utterance)
            
            single_session['log'] = log
            
            processed_data[dialogue_id] = single_session
        
        with open(saved_dir, 'w', encoding='utf-8') as fw:
            json.dump(processed_data, fw, ensure_ascii=False, indent=4)

        return processed_data


if __name__ == '__main__':
    preprocessor = PreprocessorDST()
    preprocessor.process()





            
