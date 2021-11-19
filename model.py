import numpy as np
import os
from transformers import (
    AlbertModel,
    AlbertTokenizer,
    BertModel ,
    BertPreTrainedModel ,
    BertTokenizer,
    AdamW,
)
from dataset import QADataset
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support ,
    roc_curve ,
    auc ,
    roc_auc_score ,
    precision_recall_curve,average_precision_score)
import pandas as pd
import re

class MyBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        weight = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pred = self.sigmoid(logits)
        

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(weight)
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pred,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MyALBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            weight=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pred = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(weight)
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pred,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class sequence_model(object):
    def __init__(self ,PRETRAINED_MODEL_NAME,use_multiple_gpu = False):
        
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        
        self.division_list = [
        "乳房外科", "大腸直腸外科", "婦產科", "兒科",
        "復健科", "心臟科", "感染科", "新陳代謝科",
        "泌尿科", "牙科", "皮膚科", "眼科",
        "神經內/外科", "耳鼻喉科", "胸腔科", "腎臟內科",
        "肝膽腸胃科", "血液腫瘤科", "身心科", "免疫風濕科",
        "骨科", "口腔顎面外科","一般外科"
    ]
        self.num_class = len(self.division_list)
        print("class : ",self.num_class)

        if "albert" in PRETRAINED_MODEL_NAME:
            
            self.model = MyALBertModel.from_pretrained(
                PRETRAINED_MODEL_NAME, num_labels=self.num_class
            )
        else:
            self.model = MyBertModel.from_pretrained(
                    PRETRAINED_MODEL_NAME, num_labels = self.num_class
                )
        
        if torch.cuda.is_available():
            self.model.cuda()
        if use_multiple_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.full_model = nn.DataParallel(self.model, device_ids=[0,1])
            self.full_model = self.full_model.cuda()
            self.model = self.full_model.module
        else:
          self.full_model = self.model


    def load_model(self , trained_model_version,threshold =True):
        self.model.load_state_dict(
                torch.load(
                    f"./model/model_v{trained_model_version}.pt"
                )
            )
        if threshold:
            self.threshold = pd.read_csv(f"./model/threshold_v{trained_model_version}.csv")
        self.model.cuda()


    def load_data(self,data_type ,batch_size):
        dataset = QADataset(tokenizer=self.tokenizer, mode = data_type)
        data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataset,data


    def train_validation_split(self,train_batch_size,validation_batch_size ,train_dataset, train_ratio):
        train_size = int(train_ratio * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )
        train_data = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        validation_data = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True)

        return train_data, validation_data

    def set_model(self, frezze,weight_decay, specific_lr):
        bert_param_optimizer = list(self.model.bert.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params":
                    self.model.classifier.bias
                ,
                "lr": specific_lr,
                "weight_decay": 0.0,
            },
            {
                "params":
                    self.model.classifier.weight
                ,
                "lr": specific_lr,
                "weight_decay": weight_decay,
            }
        ]

        if frezze:
            self.freeze_layers()
        # 用transformers的optimizer


    def freeze_layers(self, unfreeze_layers=None):
        if unfreeze_layers == None:
            unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break


    def train_process(self   ,epochs ,
                      warmup_steps ,
                      max_norm ,
                      model_version ,
                      early_stop = True,
                      loss_log = None ,
                      save_performance = True,
                      save_model = True,
                      weight = None):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=epochs * len(self.train_data)
        )
        if weight != None:
            weight = torch.tensor(weight, dtype=torch.long).to(device)
        print("start training")
        max_grad_norm = max_norm
        stopping = 0
        last_loss = 1e20
        last_f1 = 0
        last_AP = 0
        last_fhr = 0
        if loss_log == None:
            loss_log = {'train_loss':[],
                        'validation_loss' : []}
        else:
            loss_log = loss_log
            
        f1_list = []

        for epoch in range(epochs):

            # 把model變成train mode
            self.model.train()
            total_train_loss = 0
            train_steps_per_epoch = 0
            probability = []
            true_label = []
            for batch in tqdm(self.train_data):
                questions_ids, mask_ids, key_ids, _, _ = batch
                # 將data移到gpu
                questions_ids = questions_ids.cuda()
                mask_ids = mask_ids.cuda()
                key_ids =  key_ids.cuda()

                # 將optimizer 歸零
                self.optimizer.zero_grad()

                outputs = self.full_model(
                    questions_ids,
                    token_type_ids=None,
                    attention_mask=mask_ids,
                    labels=key_ids,
                    weight = weight
                )



                loss = outputs.loss
                output = outputs[1].detach().cpu().numpy()
                label = key_ids.detach().cpu().numpy()
                probability.extend(output)
                true_label.extend(label)
                loss.sum().backward()

                total_train_loss += loss.sum().item()
                train_steps_per_epoch += 1

                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            ##evalue threshold by pr
            true_label = np.array(true_label)
            probability = np.array(probability)
            e = 1e-20
            threshold = []
            for i in range(self.num_class):
                sub_precision, sub_recall, thresholds_list_2 = precision_recall_curve(true_label[:, i], probability[:, i])

                sub_f1_score = 2 / (1 / (sub_precision + e) + 1 / (sub_recall + e))
                index = np.argmax(sub_f1_score)
                sub_threshold = thresholds_list_2[index]
                threshold.append(sub_threshold)
            self.threshold = pd.DataFrame(threshold, columns=['threshold'], index=self.division_list)


            epoch_loss = total_train_loss / train_steps_per_epoch
            print(
                f"Epoch:{epoch + 1}/{epochs}\tTrain Loss: \
                                {epoch_loss}"
            )
            loss_log['train_loss'].append(epoch_loss)
            validation_loss ,performance,phr ,fhr ,fcr = self.validation()
            loss_log['validation_loss'].append(validation_loss)
            f1_list.append(performance['f1_score']['weighted'])
            if save_performance:
                self.save_performance(model_version = model_version,
                                      loss_log = loss_log,
                                      performance = performance,
                                      phr = phr,
                                      fhr = fhr,
                                      fcr = fcr)
            if last_loss > validation_loss:
                last_loss = validation_loss
                stopping = 0
                if save_model:
                    self.save(model_version=model_version + 'loss')
            else:
                last_loss = validation_loss
                stopping += 1

            if last_f1 < performance['f1_score']['weighted']:
                last_f1 = performance['f1_score']['weighted']
                if save_model:
                    self.save(model_version=model_version + 'f1')

            if last_AP < performance['AP']['weighted']:
                last_AP = performance['AP']['weighted']
                if save_model:
                    self.save(model_version=model_version + 'AP')

            if last_fhr < fhr:
                last_fhr = fhr
                if save_model:
                    self.save(model_version=model_version + 'fhr')


            if early_stop:
                if stopping == 2:
                    print('early stopping!')
                    break
                    
        return loss_log,performance,phr ,fhr ,fcr,f1_list


    def save_performance(self,model_version,loss_log,performance,phr ,fhr ,fcr):
        dirpath = f'./model/model_log/metrices/model_v{model_version}'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            print("Directory ", dirpath, " Created ")

        try :
            performance = pd.DataFrame(performance).applymap(lambda y: [y])
            self.performance_log = self.performance_log.apply(lambda x: x + performance[x.name])
            self.performance_log.to_csv(dirpath + '/performance.csv', encoding='utf-8-sig')

            self.loss_log = self.loss_log.append({
                'train_loss' : [loss_log['train_loss'][-1]],
                'validation_loss' : [loss_log['validation_loss'][-1]]
            } ,ignore_index = True)
            self.loss_log.to_csv(dirpath+'/loss_log.csv',encoding='utf-8-sig')

        except:
            self.performance_log = pd.DataFrame(performance).applymap(lambda y: [y])
            self.performance_log.to_csv(dirpath + '/performance.csv', encoding='utf-8-sig')

            self.loss_log = pd.DataFrame({
                'train_loss': loss_log['train_loss'],
                'validation_loss': loss_log['validation_loss']
            })
            self.loss_log.to_csv(dirpath + '/loss_log.csv', encoding='utf-8-sig')



    def train(self ,train_batch_size,learning_rate,weight_decay ,warmup_steps,
              frezze ,epochs,max_norm ,model_version ,
              early_stop = True,loss_log = None ,weight = None,save_performance = True,validation_batch_size = None,save_model = True,
              f1_return = False):
        torch.manual_seed(0)
        if validation_batch_size == None:
            validation_batch_size = train_batch_size
        self.train_dataset ,_ = self.load_data(data_type = 'train',batch_size = train_batch_size)
        self.set_model(frezze=frezze,weight_decay = weight_decay,specific_lr = learning_rate*10)
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=learning_rate)
        #self.optimizer = nn.DataParallel(self.optimizer, device_ids=[0,1])
        self.train_data, self.validation_data = self.train_validation_split(train_batch_size = train_batch_size,
                                                                            validation_batch_size = validation_batch_size,
                                                                            train_dataset = self.train_dataset,
                                                                            train_ratio=0.8)
        loss_log,performance ,phr ,fhr ,fcr,f1_list = self.train_process(epochs = epochs,
                                                                 warmup_steps = warmup_steps,
                                                                 max_norm = max_norm,
                                                                 model_version = model_version,
                                                                 early_stop = early_stop,
                                                                 loss_log = loss_log,
                                                                 save_performance = save_performance,
                                                                 weight = weight,
                                                                 save_model = save_model)
        #self.plot_loss(loss_log)
        if f1_return:
            return max(f1_list)


    def k_fold_train(self ,PRETRAINED_MODEL_NAME,batch_size,learning_rate ,
                      epochs,warmup_steps,max_norm ,model_version,weight_decay,
                      use_multiple_gpu = False ,k = 5 ,frezze = False, early_stop = True,loss_log = None):
        torch.manual_seed(123)

        self.k_fold_performance = pd.DataFrame(columns=['partially_hit_ratio',
                                                        'fully_hit_ratio',
                                                        'fully_correct_ratio',
                                                        'weighted_AUC',
                                                        'macro_AUC',
                                                        'weighted_AP',
                                                        'macro_AP',
                                                        'weighted_f1',
                                                        'macro_f1'])

        self.train_dataset, _ = self.load_data(data_type='train', batch_size=batch_size)
        split_ratio = [int(len(self.train_dataset) / k)]*(k-1) + [len(self.train_dataset) -(k-1)*int(len(self.train_dataset) / k)]
        self.k_fold_train_set = torch.utils.data.random_split(
            self.train_dataset, split_ratio
        )


        for i in range(k):

            if PRETRAINED_MODEL_NAME == "voidful/albert_chinese_tiny":
                self.model = MyALBertModel.from_pretrained(
                    PRETRAINED_MODEL_NAME, num_labels=self.num_class
                )
            else:
                self.model = MyBertModel.from_pretrained(
                    PRETRAINED_MODEL_NAME, num_labels=self.num_class
                )
            if torch.cuda.is_available():
                self.model.cuda()
            if use_multiple_gpu and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.full_model = nn.DataParallel(self.model, device_ids=[0, 1])
                self.full_model = self.full_model.cuda()
                self.model = self.full_model.module
            else:
                self.full_model = self.model
            self.set_model(frezze=frezze,weight_decay = weight_decay,specific_lr = learning_rate*10)
            self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=learning_rate)


            print(f'{i+1}/{k} fold cross validation :')
            kth_train_fold = [fold for it,fold in enumerate(self.k_fold_train_set) if it != i ]
            train_dataset = torch.utils.data.ConcatDataset(kth_train_fold)
            validation_dataset = self.k_fold_train_set[i]
            self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

            loss_log,performance,phr , fhr ,fcr,f1_list = self.train_process( epochs=epochs,
                                                                      warmup_steps=warmup_steps,
                                                                      max_norm=max_norm,
                                                                      model_version=model_version,
                                                                      early_stop=early_stop,
                                                                      loss_log=loss_log,
                                                                      save_model = False,
                                                                      save_performance=False)
            k_fold_performance = pd.DataFrame({'partially_hit_ratio' : phr ,
                                               'fully_hit_ratio' : fhr,
                                               'fully_correct_ratio' : fcr ,
                                               'weighted_AUC' : performance['AUC']['weighted'],
                                               'macro_AUC' : performance['AUC']['macro'],
                                               'weighted_AP' : performance['AP']['weighted'],
                                               'macro_AP' : performance['AP']['macro'],
                                               'weighted_f1' : performance['f1_score']['weighted'],
                                               'macro_f1' : performance['f1_score']['macro']
                                                },index = [f'{i+1}_fold'])
            self.k_fold_performance = pd.concat([self.k_fold_performance,k_fold_performance],axis = 0)
            
        self.k_fold_performance.loc['mean'] = self.k_fold_performance.mean(axis = 0)

        with pd.option_context('display.max_rows', None, 'display.max_columns',None):  # more options can be specified also
            print(self.k_fold_performance)
        self.k_fold_performance.to_csv(f'./result/model_v{model_version}_{k}_FoldTrain_TotalPerformance.csv')


    def plot_loss(self,loss_log):
        plt.plot(loss_log['train_loss'], label='Train')
        plt.plot(loss_log['validation_loss'], label='validation')
        plt.legend()
        plt.ylabel('sigmoid_binary_cross_entropy')
        plt.xlabel('epochs')
        plt.title('Loss')
        plt.show()


    def validation(self):
        self.model.eval()
        probability = []
        true_label = []
        total_loss = 0
        i = 0
        for test_batch in tqdm(self.validation_data):
            questions_ids, mask_ids, key_ids, _, full_division = test_batch
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids =  key_ids.to(device)
            with torch.no_grad():
                output = self.full_model(
                    questions_ids, token_type_ids=None, attention_mask=mask_ids ,labels = key_ids
                )
            loss = output[0].detach().cpu().numpy()
            output = output[1].detach().cpu().numpy()
            label = key_ids.detach().cpu().numpy()
            probability.extend(output)
            true_label.extend(label)
            total_loss += loss.sum()
            i += 1


        true_label = np.array(true_label)
        probability = np.array(probability)
        avg_loss = total_loss / i
        print (f"Validtaion loss : {avg_loss}")
        print('='*50)
        performance ,phr ,fhr ,fcr= self.performance_report(true_label,probability,have_threshold=True)
        print(performance)
        print(f'partially hit ratio :{phr}')
        print(f'fully hit ratio : {fhr}')
        print(f'fully correct ratio : {fcr}')
        return avg_loss,performance,phr ,fhr ,fcr


    def performance_report(self,label, prob,have_threshold = False,pred = None):
        '''
        == input ==
        label :numpy array with [n_sample ,division_num]
        prob : numpy array with [n_sample ,division_num]
        == output ==
        1.dataframe include
        AUC ,AP ,threshold ,Accuracy ,precesion ,recall ,f1 score ,support and average
        2.threshold list
        '''
        e = 1e-20
        ##AUC,AP and threshold by division
        AUC = []
        AP = []
        precision = []
        recall = []
        f1_score = []
        threshold = []
        for i in range(self.num_class):
            fpr, tpr, threshold_list_1 = roc_curve(label[:, i], prob[:, i])
            sub_precision, sub_recall, thresholds_list_2 = precision_recall_curve(label[:, i], prob[:, i])

            pr_score = average_precision_score(label[:, i], prob[:, i])
            auc1 = auc(fpr, tpr)
            sub_f1_score = 2/(1/(sub_precision+e) + 1/(sub_recall+e))
            index = np.argmax(sub_f1_score)
            sub_threshold = thresholds_list_2[index]
            '''
            index = np.argmax((tpr[1:] - fpr[1:]))
            sub_threshold = threshold_list[index + 1]
            '''

            AP.append(pr_score)
            AUC.append(auc1)
            precision.append(sub_precision[index])
            recall.append(sub_recall[index])
            f1_score.append(sub_f1_score[index])
            threshold.append(sub_threshold)

        #fix threshold calculate precision recall f1 avg
        #pred_label = [self.threshold.iloc[:, 0][self.threshold['threshold'] < b].values.tolist() for b in output]
        if have_threshold :
            print('have_threshold')
            threshold = self.threshold['threshold']
        if pred is None:
            pred = [[int(p >= t) for p, t in zip(b, threshold)] for b in prob]
        else:
            print('notice that predict result just apply in precision ,recall ,f1_score and hit ratio! ')
            pred = [[int(div in d) for div in self.division_list] for d in pred]
        report = precision_recall_fscore_support(label, pred, average=None)
        total_rep = pd.DataFrame(np.array(report).T, columns=['precision', 'recall', 'f1_score', 'support'],
                                 index=self.division_list)

        partially_hit = sum([any([i == j == 1 for i, j in zip(xi, yi)]) for xi, yi in zip(pred, label)])
        partially_hit_ratio = partially_hit/len(label)
        fully_hit = sum([all([not (i == 0 and j == 1) for i, j in zip(xi, yi)]) for xi, yi in zip(pred, label)])
        fully_hit_ratio = fully_hit / len(label)
        fully_correct = sum([all([(i == j) for i, j in zip(xi, yi)]) for xi, yi in zip(pred, label)])
        fully_correct_ratio = fully_correct / len(label)

        # avg and conbine as a dataframe
        avg = np.array([
            precision_recall_fscore_support(label, pred, average='macro'),
            precision_recall_fscore_support(label, pred, average='micro'),
            precision_recall_fscore_support(label, pred, average='weighted')
        ])
        avg_rep = pd.DataFrame(avg, index=['macro', 'micro', 'weighted'],
                               columns=['precision', 'recall', 'f1_score', 'support'])

        total_report = pd.concat([total_rep, avg_rep], axis=0)
        AUC = AUC + [roc_auc_score(label, prob, average='macro')]+\
                    [roc_auc_score(label, prob, average='micro')] +\
                    [roc_auc_score(label, prob, average='weighted')]
        AP = AP + [average_precision_score(label, prob, average='macro')]+\
                    [average_precision_score(label, prob, average='micro')] +\
                    [average_precision_score(label, prob, average='weighted')]
        AUC = pd.DataFrame(AUC, columns=['AUC'], index=self.division_list + ['macro', 'micro', 'weighted'])
        AP = pd.DataFrame(AP, columns=['AP'], index=self.division_list + ['macro', 'micro', 'weighted'])
        threshold = pd.DataFrame(threshold, columns=['threshold'], index=self.division_list)
        total_report = pd.concat([AUC ,AP, threshold,total_report], axis=1)
        return total_report ,partially_hit_ratio ,fully_hit_ratio ,fully_correct_ratio
        
    def plot_curve(self,label, prob ,performance,curve = 'PR',beta = 1):
        assert curve in ['ROC','PR']
        e = 1e-20
        colors = cm.rainbow(np.linspace(0, 1, 6))
        ##AUC,AP and threshold by division
        for i ,c in zip([1,6,7,12,17,19],colors):
            if curve == 'ROC':
                fpr, tpr, threshold_list_1 = roc_curve(label[:, i], prob[:, i])
                index1 = np.argmax((tpr[1:] - fpr[1:]))
                plt.plot(fpr, tpr, label=self.division_list[i])
                plt.scatter(fpr[index1],tpr[index1])
            else:
                sub_precision, sub_recall, thresholds_list_2 = precision_recall_curve(label[:, i], prob[:, i])
                
                sub_f1_score = 2 / (1 / ((sub_precision + e)/(beta+1)) + beta / ((sub_recall + e)/(beta+1)))
                index2 = np.argmax(sub_f1_score)
                plt.plot(sub_recall[::-1],sub_precision[::-1],  label=self.division_list[i],c = c)
                plt.scatter(sub_recall[index2] ,sub_precision[index2] ,marker = "o",c = c)
                plt.scatter(performance['recall'][self.division_list[i]] ,performance['precision'][self.division_list[i]] ,marker = 'x',c = c)
        
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(curve + 'curve')
        font = font_manager.FontProperties(family="SimSun",
                                   weight='bold',
                                   style='normal', size=10)
        plt.legend(prop=font)
        plt.savefig(curve + 'curve')


    def save(self,model_version):
        torch.save(self.model.state_dict(), f"./model/model_v{model_version}.pt")
        self.threshold.to_csv(f"./model/threshold_v{model_version}.csv")


    def test(self,batch_size, save=True, result_data_name='result', rule=False):
        _, test_data = self.load_data(data_type='test',batch_size = batch_size)
        self.model.eval()
        true_label = []
        probability = []
        predict_label = []
        total_loss = 0
        i = 0
        total_data = pd.DataFrame(columns=['Question', 'True_division', 'Pred_division'])

        for test_batch in tqdm(test_data):
            questions_ids, mask_ids, key_ids, full_question, full_division = test_batch
            questions_ids = questions_ids.cuda()
            mask_ids = mask_ids.cuda()
            key_ids = key_ids.cuda()
            with torch.no_grad():
                output = self.model(
                    questions_ids, token_type_ids=None, attention_mask=mask_ids, labels=key_ids
                )
            loss = output[0].detach().cpu().numpy()
            output = output[1].detach().cpu().numpy()
            prob = output
            pred_label = [self.threshold.iloc[:, 0][self.threshold['threshold'] < b].values.tolist() for b in output]
            if rule:
                rule_pred = []
                for ans in pred_label:
                    ans = self.rule(ans)
                    rule_pred.append(ans)
                pred_label = rule_pred
            label = key_ids.detach().cpu().numpy()
            total_data = total_data.append(pd.DataFrame({'Question': list(full_question),
                                            'True_division': np.array([full_division]).T.reshape(-1,24).tolist(),
                                            'Pred_division': pred_label}), ignore_index=True)
            probability.extend(prob)
            true_label.extend(label)
            predict_label.extend(pred_label)
            total_loss += loss
            i += 1

        true_label = np.array(true_label)
        probability = np.array(probability)
        predict_label = np.array(predict_label)
        avg_loss = total_loss / i
        print(f"Validtaion loss : {avg_loss}")
        print('=' * 50)
        if rule:
            performance, _, phr, fhr, fcr = self.performance_report(true_label, probability, have_threshold=True,
                                                                    pred=predict_label)
        else:
            performance, _, phr, fhr, fcr = self.performance_report(true_label, probability, have_threshold=True)
        print(performance)
        print(f'partially hit ratio :{phr}')
        print(f'fully hit ratio : {fhr}')
        print(f'fully correct ratio : {fcr}')
        if save:
            total_data.to_csv(f'./result/{result_data_name}.csv', encoding='utf-8-sig')
    
    
    def test_for_dataloader(self,dataloader, save=True, result_data_name='result', rule=False,curve = None ,beta = 1):
        self.model.eval()
        true_label = []
        probability = []
        predict_label = []
        total_loss = 0
        i = 0
        total_data = pd.DataFrame(columns=['Question', 'True_division', 'Pred_division' ,'probability'])

        for test_batch in tqdm(dataloader):
            questions_ids, mask_ids, key_ids, full_question, full_division = test_batch
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids = key_ids.to(device)
            with torch.no_grad():
                output = self.model(
                    questions_ids, token_type_ids=None, attention_mask=mask_ids, labels=key_ids
                )
            loss = output[0].detach().cpu().numpy()
            output = output[1].detach().cpu().numpy()
            prob = output
            pred_label = [self.threshold.iloc[:, 0][self.threshold['threshold'] < b].values.tolist() for b in output]
            if rule:
                rule_pred = []
                for ans in pred_label:
                    ans = self.rule(ans)
                    rule_pred.append(ans)
                pred_label = rule_pred
            label = key_ids.detach().cpu().numpy()
            total_data = total_data.append(pd.DataFrame({'Question': list(full_question),
                                            'True_division': np.array([full_division]).T.reshape(-1,24).tolist(),
                                            'Pred_division': pred_label,
                                            'probability' : prob.tolist()}), ignore_index=True)
            probability.extend(prob)
            true_label.extend(label)
            predict_label.extend(pred_label)
            total_loss += loss
            i += 1

        true_label = np.array(true_label)
        probability = np.array(probability)
        predict_label = np.array(predict_label)
        avg_loss = total_loss / i
        print(f"Validtaion loss : {avg_loss}")
        print('=' * 50)
        if rule:
            performance, _, phr, fhr, fcr = self.performance_report(true_label, probability, have_threshold=True,
                                                                    pred=predict_label)
        else:
            performance, _, phr, fhr, fcr = self.performance_report(true_label, probability, have_threshold=True)
        print(performance)
        print(f'partially hit ratio :{phr}')
        print(f'fully hit ratio : {fhr}')
        print(f'fully correct ratio : {fcr}')
        if save:
            total_data.to_csv(f'./result/{result_data_name}.csv', encoding='utf-8-sig')
            
        if curve != None:
            self.plot_curve(label = true_label,
                            prob =  probability,
                            performance = performance,
                            curve = curve,
                            beta = beta)


    def test_for_input(self ,question ,output_prob = False ,rule_base = False ,child = False):
        if len(question) > 512:
            question = question[: 512 - 2]

        # 統一逗號
        question = re.sub(",", "，", question)
        question = re.sub("？", "?", question)
        question = re.sub("、", "，", question)
        # 消除多餘空白
        question = re.sub(" ", "", question)
        question = re.sub("\\s+", "", question)
        # 刪除多餘換行
        question = re.sub("\r*\n*", "", question)
        # 刪除括號內英文
        question = re.sub("\\([a-z A-Z \\-]*\\)", "", question)

        question_token = ["[CLS]"] + self.tokenizer.tokenize(question) + ["[SEP]"]


        question_token = question_token + (512 - len(question_token)) * ["[PAD]"]
        question_ids = self.tokenizer.convert_tokens_to_ids(
            question_token
        )
        mask_ids = [float(i > 0) for i in question_ids]

        self.model.eval()
        with torch.no_grad():
            question_ids = torch.tensor(
                [question_ids], dtype=torch.long).cuda()
            mask_ids = torch.tensor([mask_ids], dtype=torch.long).cuda()

            output = self.model(
                question_ids, token_type_ids=None, attention_mask=mask_ids)
        output = output[0].detach().cpu().numpy()
        if output_prob:
            for d in range(len(self.division_list)):
                print(self.division_list[d] , ':' , output[0][d])
            return output[0]

        pred_label = [self.threshold.iloc[:,0][self.threshold['threshold'] < b].values.tolist() for b in output]
        if rule_base:
            for i in range(len(pred_label)):
                pred_label[i] = self.rule(pred_label[i] ,child)

        return pred_label


    def rule(self ,answer ,child = False):
        self.group_dict = dict([
            ("一般外科", "0"), ("乳房外科", "1"), ("大腸直腸外科", "1"), ("婦產科", "3"),
            ("家醫科", "0"), ("兒科", "0"), ("復健科", "3"), ("心臟科", "2"), ("感染科", "2"),
            ("新陳代謝科", "2"), ("泌尿科", "3"), ("牙科", "3"), ("皮膚科", "3"), ("眼科", "3"),
            ("神經內科", "3"), ("神經外科", "3"), ("耳鼻喉科", "3"), ("胸腔科", "3"),
            ("腎臟內科", "2"), ("肝膽腸胃科", "2"), ("血液腫瘤科", "2"), ("身心科", "3"),
            ("免疫風濕科", "2"), ("骨科", "3"), ("神經內/外科", "3"), ("中醫科", "3"),
            ("口腔顎面外科", "1"), ("一般內科", "0")
        ])
        self.Chinese_medicine_1 = dict([
            ("一般外科", "0"), ("乳房外科", "0"), ("大腸直腸外科", "0"), ("婦產科", "1"),
            ("家醫科", "0"), ("兒科", "0"), ("復健科", "1"), ("心臟科", "0"), ("感染科", "0"),
            ("新陳代謝科", "0"), ("泌尿科", "0"), ("牙科", "0"), ("皮膚科", "2"), ("眼科", "0"),
            ("神經內科", "0"), ("神經外科", "0"), ("耳鼻喉科", "0"), ("胸腔科", "0"),
            ("腎臟內科", "0"), ("肝膽腸胃科", "1"), ("血液腫瘤科", "0"), ("身心科", "0"),
            ("免疫風濕科", "1"), ("骨科", "2"), ("神經內/外科", "2"), ("中醫科", "0"),
            ("口腔顎面外科", "0"), ("一般內科", "0")
        ])
        self.Chinese_medicine_2 = dict([
            ("一般外科", ["0"]), ("乳房外科", ["0"]), ("大腸直腸外科", ["0"]), ("婦產科", ["0"]),
            ("家醫科", ["0"]), ("兒科", ["0"]), ("復健科", ["骨科", "神經內/外科"]), ("心臟科", ["0"]), ("感染科", ["0"]),
            ("新陳代謝科", ["0"]), ("泌尿科", ["0"]), ("牙科", ["0"]), ("皮膚科", ["免疫風濕科"]), ("眼科", ["0"]),
            ("神經內科", ["0"]), ("神經外科", ["0"]), ("耳鼻喉科", ["0"]), ("胸腔科", ["0"]),
            ("腎臟內科", ["0"]), ("肝膽腸胃科", ["0"]), ("血液腫瘤科", ["0"]), ("身心科", ["0"]),
            ("免疫風濕科", ["0"]), ("骨科", ["復健科"]), ("神經內/外科", ["復健科"]), ("中醫科", ["0"]),
            ("口腔顎面外科", ["0"]), ("一般內科", ["0"])
        ])

        if len(answer) == 0:  # 都沒有的新增家醫科
            answer.append('家醫科')
            if child is True:
                answer.append('兒科')
        elif len(answer) == 1:
            if self.group_dict[answer[0]] == "1":  # 只有一個外科的新增一般外科
                answer.append('一般外科')
                if child is True:  # 不到18新增小兒科
                    answer.append('兒科')
            elif self.group_dict[answer[0]] == "2":  # 只有一個內科的新增一般內科
                answer.append('一般內科')
                if child is True:
                    answer.append('兒科')
            elif self.group_dict[answer[0]] == "3":  # 只有一個不是內科也不是外科，新增家醫科
                answer.append('家醫科')
                if child is True:
                    answer.append('兒科')
        elif len(answer) == 2:
            if child is True:  # 未滿18， 直接新增小兒科
                answer.append('兒科')
            elif (
                    self.group_dict[answer[0]] == "0" and self.group_dict[answer[1]] == "0"
            ):
                pass
            elif (
                    self.group_dict[answer[0]] == "1" and self.group_dict[answer[1]] == "1"
            ):
                answer.append('一般外科')
            elif (
                    self.group_dict[answer[0]] == "2" and self.group_dict[answer[1]] == "2"
            ):
                answer.append('一般內科')
            elif (
                    self.group_dict[answer[0]] == "3" and self.group_dict[answer[1]] == "3"
            ):
                answer.append('家醫科')
            elif (
                    self.group_dict[answer[0]] != self.group_dict[answer[1]] and '家醫科' not in answer
            ):
                answer.append('家醫科')  ##

            ##中醫 rule base
            elif '中醫科' in self.division_list:
                if (
                        self.Chinese_medicine_1[answer[0]] == "1" or self.Chinese_medicine_1[answer[1]] == "1"
                ):
                    answer.append('中醫科')
                elif (
                        self.Chinese_medicine_2[answer[0]] != "0" or
                        self.Chinese_medicine_2[answer[0]] != "0"
                ):
                    answer.append('中醫科')
        if len(answer) == 0:
            answer.append('家醫科')
        return answer





