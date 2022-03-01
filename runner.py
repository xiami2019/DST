import os
import glob
import copy
import time
import math
import torch
import shutil
from abc import ABCMeta, abstractmethod
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, BartForConditionalGeneration, TFGPT2LMHeadModel
from modeling_cpt import CPTForConditionalGeneration
from reader import CrossWOZReader, CrossWOZIterator
from evaluator import CrossWOZEvaluator
from utils import definitions_cw
from utils.io_utils import get_or_create_logger, save_json, save_pickle

logger = get_or_create_logger(__name__)

class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]

            do_span_stats = True
        else:
            do_span_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_span_stats, do_resp_stats)

    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        logger.info(
            " ".join([common_info, belief_info, resp_info, span_info]))

        self.init_stats()

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader) -> None:
        self.cfg = cfg
        self.reader = reader
        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
        else:
            model_path = self.cfg.backbone
        logger.info('Load model from {}'.format(model_path))

        if self.cfg.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large']:
            model = CPTForConditionalGeneration.from_pretrained(model_path)
        elif self.cfg.backbone in ['fnlp/bart-base-chinese', 'fnlp/bart-large-chinese']:
            model = BartForConditionalGeneration.from_pretrained(model_path)
        elif self.cfg.backbone in ['mymusise/CPM-GPT2-FP16', 'mymusise/gpt2-medium-chinese']:
            model = TFGPT2LMHeadModel.from_pretrained(model_path)
        else:
            raise NotImplementedError
        
        model.resize_token_embeddings(self.reader.vocab_size)
        model.to(self.cfg.device)

        return model

    def save_model(self, epoch):
        latest_ckpt = 'ckpt-epoch{}'.format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
        model = self.model
        model.save_pretrained(save_path)
        self.reader.tokenizer.save_pretrained(save_path)

        # keep checkpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, 'ckpt-*')),
            key=os.path.getmtime,
            reverse=True
        )
        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_training_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_training_steps_per_epoch * self.cfg.epochs) // self.cfg.grad_accum_steps
        
        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))
        
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        num_count = label.view(-1).ne(pad_id).long().sum()
        num_correct = 0
        for i in range(label.shape[0]):
            one_pred = pred[i]
            one_label = label[i]
            valid_len = one_label.ne(pad_id).long().sum()
            one_pred = one_pred[:valid_len]
            one_label = one_label[:valid_len]
            num_correct += (one_pred == one_label).sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

class CrossWOZRunner(BaseRunner):
    def __init__(self, cfg) -> None:
        reader = CrossWOZReader(cfg)
        self.iterator = CrossWOZIterator(reader)
        super().__init__(cfg, reader)

    def step_fn(self, inputs, belief_labels):
        inputs = inputs.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=belief_labels,
        )

        belief_loss = belief_outputs[0]
        belief_logits = belief_outputs[1]
        belief_pred = torch.argmax(belief_logits, dim=-1)

        num_belief_correct, num_belief_count = self.count_tokens(belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if num_belief_correct > num_belief_count:
            raise Exception('acc calculating error')

        loss = belief_loss

        step_outputs = {
            'belief': {
                'loss': belief_loss.item(),
                'correct': num_belief_correct.item(),
                'count': num_belief_count.item(),
            }
        }

        return loss, step_outputs
    
    def train_epoch(self, train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter=None):
        self.model.train()
        self.model.zero_grad()

        with tqdm(total=num_training_steps_per_epoch) as pbar:
            for step, batch in enumerate(train_iterator):
                start_time = time.time()
                inputs, belief_labels = batch

                loss, step_outputs = self.step_fn(inputs, belief_labels)

                if self.cfg.grad_accum_steps > 1:
                    loss = loss / self.cfg.grad_accum_steps
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm)

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    lr = scheduler.get_last_lr()[0]

                    if reporter is not None:
                        reporter.step(start_time, lr, step_outputs)
                pbar.update(1)

    def train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            'train', self.cfg.batch_size, shuffle=True, num_dialogs=self.cfg.num_train_dialogs,
            special_domain=self.cfg.special_domain
        )

        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch, self.cfg.batch_size)
        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.context_size
            )

            self.train_epoch(train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter)

            logger.info('done {}/{} epoch'.format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            self.validation(reporter.global_step)

    def validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            'val', self.cfg.batch_size, special_domain=self.cfg.special_domain
        )

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.context_size
        )

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc='Validation'):
            start_time = time.time()
            inputs, belief_labels = batch
            _, step_outputs = self.step_fn(inputs, belief_labels)
            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_span_stats = True if "span" in step_outputs else False
        do_resp_stats = True if "resp" in step_outputs else False
        reporter.info_stats("val", global_step, do_span_stats, do_resp_stats)

        torch.set_grad_enabled(True)

    def finalize_bspn(self, belief_outputs):
        eos_token_id = self.reader.get_token_id(definitions_cw.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_outzput = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}
            decoded['bspn_gen'] = bspn
            batch_decoded.append(decoded)
        return batch_decoded

    def predict(self):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size, special_domain=self.cfg.special_domain
        )

        early_stopping = True if self.cfg.beam_size > 1 else False

        results = {}
        count = 0
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc='Prediction'):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn['user']), self.cfg.context_size
                    )

                    encoder_input_ids = context + turn['user'] + [self.reader.eos_token_id]
                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)
                attention_mask = torch.where(batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                # belief tracking
                with torch.no_grad():
                    belief_outputs = self.model.generate(
                        input_ids=batch_encoder_input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=200,
                    )

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                decoded_belief_outputs = self.finalize_bspn(belief_outputs)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn['user'])

                    # use true previous belief states and ignore the db stats
                    pv_bspn = turn['bspn']
                    
                    # use true previous action
                    pv_aspn = turn['aspn']

                    # use true previous response
                    pv_resp = turn['resp']

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        if self.cfg.use_true_bs:
                            pv_text += (pv_bspn + pv_resp)
                        else:
                            pv_text += (belief_outputs[0][1:-1] + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))
        
        evaluator = CrossWOZEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == 'dst':
            metric_results = {}

            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(results)
            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))
            metric_results['all'] = {'joint acc': joint_goal, 'acc': accuracy, 'f1': f1}
            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100
                metric_results[domain_slot] = acc
                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
            save_json(results, os.path.join(self.cfg.ckpt, 'metric_results'))
        else:
            raise NotImplementedError


                    
