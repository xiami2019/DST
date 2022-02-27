import pprint
import os
from utils import definitions_cw
from utils.io_utils import get_or_create_logger, load_json

logger = get_or_create_logger(__name__)

class CrossWOZEvaluator(object):
    def __init__(self, reader, eval_data_type="test"):
        self.reader = reader
        self.all_domains = definitions_cw.ALL_DOMAINS

        self.gold_data = load_json(os.path.join(
            self.reader.data_dir, "{}_mttod.json".format(eval_data_type)))

        self.eval_data_type = eval_data_type

        self.all_info_slot = []
        for d, s_list in definitions_cw.INFORMABLE_SLOTS.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

    def value_similar(self, a, b):
        return True if a == b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn):
        constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s, v in cons.items():
                key = domain+'-'+s
                constraint_dict_flat[key] = v

        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons,
                            slot_appear_num=None, slot_correct_num=None):
        tp, fp, fn = 0, 0, 0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            # v_truth = truth_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(
                        slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(
                    slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp, fp, fn, acc, list(set(false_slot))

    def dialog_state_tracking_eval(self, dials):
        total_turn, joint_match = 0, 0
        total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num, turn in enumerate(dial):
                gen_cons = self._bspn_to_dict(turn['bspn_gen'])
                truth_cons = self._bspn_to_dict(turn['bspn'])

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                tp, fp, fn, acc, false_slots = self._constraint_compare(
                    truth_cons, gen_cons, slot_appear_num, slot_correct_num)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / \
            (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100

        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num

    def aspn_eval(self, dials, eval_dial_list=None):
        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1 * 100

    def context_to_response_eval(self, dials, eval_dial_list=None, add_auxiliary_task=False):
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}

            for domain in self.all_domains:
                if self.gold_data[dial_id]['goal'].get(domain):
                    true_goal = self.gold_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(
                dial, goal, reqs, counts, add_auxiliary_task=add_auxiliary_task)
            '''
            if success == 0 or match == 0:
                print("success ", success, "; match ", match)
                print(goal)
                for turn in dial:
                    print("=" * 50 + " " + str(dial_id) + " " + "=" * 50)
                    print("user               | ", turn["user"])
                    print("-" * 50 + " " + str(turn["turn_num"]) + " " + "-" * 50)
                    print("bspn               | ", turn["bspn"])
                    print("bspn_gen           | ", turn["bspn_gen"])
                    if "bspn_gen_with_span" in turn:
                        print("bspn_gen_with_span | ", turn["bspn_gen_with_span"])
                    print("-" * 100)
                    print("resp               | ", turn["redx"])
                    print("resp_gen           | ", turn["resp_gen"])
                    print("=" * 100)

                input()
            '''
            successes += success
            matches += match
            dial_num += 1

            # for domain in gen_stats.keys():
            #     gen_stats[domain][0] += stats[domain][0]
            #     gen_stats[domain][1] += stats[domain][1]
            #     gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
            #     for domain in gen_stats.keys():
            #         sng_gen_stats[domain][0] += stats[domain][0]
            #         sng_gen_stats[domain][1] += stats[domain][1]
            #         sng_gen_stats[domain][2] += stats[domain][2]

        # self.logger.info(report)
        succ_rate = successes/(float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100

        return succ_rate, match_rate, counts, dial_num

    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                   soft_acc=False, add_auxiliary_task=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
        #'id'
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0:
                continue

            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if add_auxiliary_task:
                            bspn = turn['bspn_gen_with_span']
                        else:
                            bspn = turn['bspn_gen']

                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspn_to_constraint_dict(
                            bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(
                                domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if ven not in venue_offered[domain]:
                                    # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:
                                # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            # if pointer was allowing for that?
                            if 'booked' in turn['pointer'] or 'ok' in turn['pointer']:
                                provided_requestables[domain].append(
                                    'reference')
                            # provided_requestables[domain].append('reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0],
                'hotel': [0, 0, 0],
                'attraction': [0, 0, 0],
                'train': [0, 0, 0],
                'taxi': [0, 0, 0],
                'hospital': [0, 0, 0],
                'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(
                    domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]) is str and \
                   '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and \
                     len(set(venue_offered[domain]) & set(goal_venues))>0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for reqs in true_goal[domain]['reqt']:  # addtional requests:
                        if reqs in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(reqs)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append("reference")

            for s, v in true_goal[domain]['info'].items():
                s_, v_ = clean_slot_values(domain, s, v)
                if len(v_.split()) >1:
                    v_ = ' '.join(
                        [token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]["informable"][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']

        return goal

    def run_metrics(self, data, domain="all", file_list=None):
        metric_result = {'domain': domain}

        bleu = self.bleu_metric(data, file_list)

        jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(
            data, file_list)

        metric_result.update(
            {'joint_goal': jg, 'slot_acc': slot_acc, 'slot_f1': slot_f1})

        info_slots_acc = {}
        for slot in slot_cnt:
            correct = slot_corr.get(slot, 0)
            info_slots_acc[slot] = correct / slot_cnt[slot] * 100
        info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key=lambda x: x[1]))

        act_f1 = self.aspn_eval(data, file_list)

        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, file_list)

        req_slots_acc = {}
        for req in self.requestables:
            acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
            req_slots_acc[req] = acc * 100
        req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

        if dial_num:
            metric_result.update({'act_f1': act_f1,
                'success': success,
                'match': match,
                'bleu': bleu,
                'req_slots_acc': req_slots_acc,
                'info_slots_acc': info_slots_acc,
                'dial_num': dial_num})

            logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f',
                         jg, slot_acc, slot_f1, act_f1)
            logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f',
                         match, success, bleu)
            logging.info('[CTR] ' + '; '
                         .join(['%s: %2.1f' %(req, acc) for req, acc in req_slots_acc.items()]))

            return metric_result
        else:
            return None

    def e2e_eval(self, data, eval_dial_list=None, add_auxiliary_task=False):
        bleu = self.bleu_metric(data)
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=add_auxiliary_task)

        return bleu, success, match