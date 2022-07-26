import os
import math
import numpy as np

from sklearn import multiclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics.functional 

def train(args, model, optimizer, trainset, validset, scheduler=None, logger=None):
    if args.mode.endswith('_bce'):
        train_point_bce(args, model, optimizer, trainset, validset, scheduler, logger)

    elif args.mode.endswith('_ce'):
        train_point_ce(args, model, optimizer, trainset, validset, scheduler, logger)

    elif args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type_attn', 'seq2seq_type_new',]:
        train_type(args, model, optimizer, trainset, validset, scheduler, logger)

    elif args.mode in ['single_point', 'single_point_attn', 'seq2seq_point', 'seq2seq_point_attn']:
        train_point(args, model, optimizer, trainset, validset, scheduler, logger)

    elif args.mode in ['single_all', 'single_all_attn', 'seq2seq_all', 'seq2seq_all_attn']:
        train_all(args, model, optimizer, trainset, validset, scheduler, logger)

    else:
        raise Exception(f"Not defined mode: {args.mode}")



def evaluate(args, model, dataset):
    if args.mode.endswith('ce'):
        return eval_point_ce(args, model, dataset)
    elif args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type', 'seq2seq_type_attn']:
        return eval_type(args, model, dataset)
    
    elif args.mode in ['single_point', 'single_point']:
        return eval_point(args, model, dataset)
    

    else:
        raise Exception(f"Not defined mode: {args.mode}")


def train_type(args, model, optimizer, trainset, validset, scheduler, logger):
    
    best_acc = 0.
    for epoch in range(args.epochs):
        
        print(f"[{epoch}/{args.epochs}] epochs training...")
        train_loss, acc_type = 0., 0
        model.train()        
        for i, (pre, post, label_type, _, _) in enumerate(trainset, 1):
            # print(label_type.shape)
            label_type = label_type.to(args.device)
            out = model(pre.to(args.device), post.to(args.device), label_type)
            
            label = label_type.reshape(-1)
            # print(out.reshape(-1, args.vocab_size).shape)
            loss = F.cross_entropy(out.reshape(-1, args.vocab_size), label, reduction='mean')

            predicted = out.argmax(-1).reshape(-1)


            # total += label.size(0)
            # print(predicted.shape, label.shape, predicted.eq(label))
            # correct += predicted.eq(label).sum().item()
            train_loss += loss.data

            # print(correct/total, correct, total)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            acc_type += predicted.eq(label).sum().item() / label.size(0)
            
        
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / i,
                # 'train/ppl': math.exp(train_loss / (i+1)),
                'train/acc_type': 100. * acc_type / i 
            })
        
        if epoch % args.valid_interval == 0:
            valid_loss, valid_acc = eval_type(args, model, validset)
            print('[epoch %d/%d] val_loss: %.3f' %
                (epoch, args.epochs, valid_loss))            
            print('[epoch %d/%d] val_acc: %.3f' %
                (epoch, args.epochs, valid_acc))

                
            if best_acc < valid_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), 
                                    os.path.join(args.output_dir, 'best.pt') )
            
            if logger is not None:
                logger.log({
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'valid/loss': valid_loss,
                    # 'train/ppl': math.exp(train_loss / (i+1)),
                    'valid/acc_type': valid_acc 
                })    
        print(f'Epoch : {epoch} Done!')        
        if scheduler is not None:
            scheduler.step()


def eval_type(args, model, dataset):
    model.eval()
            
    valid_loss, acc_type = 0., 0
    for i, (pre, post, label_type, _, _) in enumerate(dataset, 1):
        with torch.no_grad():

            label_type = label_type.to(args.device)
            out = model(pre.to(args.device), post.to(args.device), label_type, teacher_forcing_ratio=0)

            label = label_type.reshape(-1)
            loss = F.cross_entropy(out.reshape(-1, args.vocab_size), label, reduction='mean')

            predicted = out.argmax(-1).reshape(-1)
            
            acc_type += predicted.eq(label).sum().item() / label.size(0)
            

            # total += label.size(0)
            # # print(predicted.shape, label.shape, predicted.eq(label))
            # correct += predicted.eq(label).sum().item()
            valid_loss += loss.data
    
    valid_loss /= i
    valid_acc = 100. * acc_type / i
    return valid_loss, valid_acc


def train_point_bce(args, model, optimizer, trainset, validset, scheduler, logger):
    
    best_f1 = 0.
    for epoch in range(args.epochs):
        print(f"[{epoch}/{args.epochs}] epochs training...")

        train_loss, acc_point, train_f1 = 0., 0., 0.
        train_precision, train_recall = 0., 0.
        model.train()        
        for i, (pre, post, label_type, label_pre, label_post) in enumerate(trainset, 1):
            assert args.trg_len == len(label_type) == len(label_pre)

            # print(label_pre.argmax(-1))    
            label_pre = label_pre.to(args.device)
            label_post = label_post.to(args.device)
        

            # bsz, seq_len, 
            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            # print(pre_out.shape, label_pre.shape)
            
            
            loss = (F.cross_entropy(pre_out.reshape(-1, 2), label_pre.reshape(-1)) + \
                    F.cross_entropy(post_out.reshape(-1, 2), label_post.reshape(-1)) ) / 2
            # same with nn.CrossEntropyLoss(pre_out, label_pre)
            # loss = -1/2 * ( F.log_softmax(pre_out, dim=-1).gather(-1, label_pre.unsqueeze(-1)).mean() + \
            #               F.log_softmax(post_out, dim=-1).gather(-1, label_post.unsqueeze(-1)).mean()) 
            


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            predicted_pre = pre_out.argmax(-1).reshape(-1)
            predicted_post = post_out.argmax(-1).reshape(-1)

            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)
            
            acc_point += ( predicted_pre.eq(label_pre).sum().item() / predicted_pre.size(0) + 
                        predicted_post.eq(label_post).sum().item() / predicted_post.size(0) ) /2.


            precision_pre, recall_pre = torchmetrics.functional.precision_recall(predicted_pre, label_pre.int(), multiclass=False)
            precision_post, recall_post = torchmetrics.functional.precision_recall(predicted_post, label_post.int(), multiclass=False)
            
            precision_batch = (precision_pre + precision_post)/2.
            recall_batch = (recall_pre + recall_post)/2.
            f1_score = 2.0 / (1/precision_batch + 1/recall_batch)

            train_loss += loss.data
            train_f1 += f1_score.data
            train_precision += precision_batch.data
            train_recall += recall_batch.data


        # if logger is not None and i % args.train_interval:
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / i,
                'train/f1': train_f1 / i,
                'train/precision': train_precision / i,
                'train/recall': train_recall / i,
                'train/acc_point': 100. * acc_point / i 
            })

        if epoch % args.valid_interval == 0:

            valid_stats = eval_point_bce(args, model, validset)
            print('[epoch %d/%d] val_loss: %.3f' %
                (epoch, args.epochs, valid_stats['loss']))            
            print('[epoch %d/%d] val_acc: %.3f' %
                (epoch, args.epochs, valid_stats['acc_point']))  
            print('[epoch %d/%d] val_f1: %.3f' %
                (epoch, args.epochs, valid_stats['f1']))

                
            if best_f1 < valid_stats['f1']:
                best_f1 = valid_stats['f1']
                torch.save(model.state_dict(), 
                                    os.path.join(args.output_dir, 'best.pt') )
            
            if logger is not None:
                logger.log({
                    'epoch': epoch,
                    'best_f1': best_f1,
                    **{f"valid/{k}":v for k,v in valid_stats.items()}
                })
        print(f'Epoch : {epoch} Done!')
        if scheduler is not None:
            scheduler.step()



def eval_point_bce(args, model, dataset):
    model.eval()

    f1 = []
    valid_loss, valid_f1, acc_point = 0., 0., 0
    valid_precision, valid_recall = 0., 0.
    for i, (pre, post, label_type, label_pre, label_post) in enumerate(dataset, 1):
        with torch.no_grad():

            label_pre = label_pre.to(args.device)
            label_post = label_post.to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type, teacher_forcing_ratio=0)

            
            loss = (F.cross_entropy(pre_out.reshape(-1, 2), label_pre.reshape(-1)) + \
                    F.cross_entropy(post_out.reshape(-1, 2), label_post.reshape(-1)) ) / 2
            # same with nn.CrossEntropyLoss(pre_out, label_pre)
            # loss = -1 * ( F.log_softmax(pre_out, dim=-1).gather(-1, label_pre.unsqueeze(-1)).mean() + \
            #               F.log_softmax(post_out, dim=-1).gather(-1, label_post.unsqueeze(-1)).mean())

            predicted_pre = pre_out.argmax(-1).reshape(-1)
            predicted_post = post_out.argmax(-1).reshape(-1)
            
            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)

            acc_point += (predicted_pre.eq(label_pre).sum().item() / predicted_pre.size(0) + 
                        predicted_pre.eq(label_pre).sum().item() / predicted_post.size(0) ) /2.
            

            precision_pre, recall_pre = torchmetrics.functional.precision_recall(predicted_pre, label_pre.int(), multiclass=False)
            precision_post, recall_post = torchmetrics.functional.precision_recall(predicted_post, label_post.int(), multiclass=False)
            
            precision_batch = (precision_pre + precision_post)/2.
            recall_batch = (recall_pre + recall_post)/2.
            f1_score = 2.0 / (1/precision_batch + 1/recall_batch)


            valid_loss += loss.data
            valid_f1 += f1_score.data
            valid_precision += precision_batch.data
            valid_recall += recall_batch.data



    valid_loss /= i
    valid_acc = 100. * acc_point / i
    valid_f1 = valid_f1 / i
    valid_precision = valid_precision / i
    valid_recall = valid_recall / i

    return {'loss': valid_loss, 'acc_point': valid_acc, 'f1': valid_f1, 'precision': valid_precision, 'recall': valid_recall}




def train_point_ce(args, model, optimizer, trainset, validset, scheduler, logger):
    
    
    best_f1 = 0.
    for epoch in range(args.epochs):
        
        print(f"[{epoch}/{args.epochs}] epochs training...")
        train_loss, acc_point, train_f1 = 0., 0., 0.
        train_precision, train_recall = 0., 0.
        model.train()        
        for i, (pre, post, label_type, label_pre, label_post) in enumerate(trainset, 1):
            assert args.trg_len == len(label_type) == len(label_pre)
            

            label_pre = label_pre.to(args.device)
            label_post = label_post.to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            
            len_pre = label_pre.size(-1) 
            loss = (F.cross_entropy(pre_out.reshape(-1, len_pre), label_pre.argmax(-1).reshape(-1)) + \
                    F.cross_entropy(post_out.reshape(-1, len_pre), label_post.argmax(-1).reshape(-1)) ) / 2

            # same with cross entropy
            # log_prob_pre = F.log_softmax(pre_out, dim=-1) * label_pre
            # log_prob_post = F.log_softmax(post_out, dim=-1) * label_post
            # loss = -1 * (log_prob_pre.mean(-1) + log_prob_post.mean(-1)).mean()
        
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()


            # predicted_pre = pre_out.argmax(-1)
            # predicted_post = post_out.argmax(-1)
            # label_pre = label_pre.argmax(-1)
            # label_post = label_post.argmax(-1)


            predicted_pre = F.one_hot(pre_out.argmax(-1), num_classes=pre_out.size(-1)).reshape(-1)
            predicted_post = F.one_hot(post_out.argmax(-1), num_classes=post_out.size(-1)).reshape(-1)

            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)

            
            acc_point += ( predicted_pre.eq(label_pre).sum().item() / predicted_pre.size(0) + 
                        predicted_post.eq(label_post).sum().item() / predicted_post.size(0) ) /2.

            precision_pre, recall_pre = torchmetrics.functional.precision_recall(predicted_pre, label_pre.int(), multiclass=False)
            precision_post, recall_post = torchmetrics.functional.precision_recall(predicted_post, label_post.int(), multiclass=False)
            train_precision += (precision_pre+precision_post)/2.
            train_recall += (recall_pre+recall_post)/2.

            
            train_f1 += (torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False) \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)) / 2

            train_loss += loss.data
        
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / i,
                'train/f1': train_f1 / i,
                'train/precision': train_precision / i,
                'train/recall': train_recall / i,
                'train/acc_point': 100. * acc_point / i 
            })
        
        if epoch % args.valid_interval == 0:
            valid_stats = eval_point_ce(args, model, validset)
            print('[epoch %d/%d] val_loss: %.3f' %
                (epoch, args.epochs, valid_stats['loss']))            
            print('[epoch %d/%d] val_acc: %.3f' %
                (epoch, args.epochs, valid_stats['acc_point']))  
            print('[epoch %d/%d] val_f1: %.3f' %
                (epoch, args.epochs, valid_stats['f1']))           

                
            if best_f1 < valid_stats['f1']:
                best_f1 = valid_stats['f1']
                torch.save(model.state_dict(), 
                                    os.path.join(args.output_dir, 'best.pt') )
            
            if logger is not None:
                logger.log({
                    'epoch': epoch,
                    'best_f1': best_f1,
                    **{f"valid/{k}":v for k,v in valid_stats.items()}
                })    
        print(f'Epoch : {epoch} Done!')
        if scheduler is not None:
            scheduler.step()



def eval_point_ce(args, model, dataset):
    model.eval()

    
    valid_loss, valid_f1, acc_point = 0., 0., 0
    valid_precision, valid_recall = 0., 0.
    for i, (pre, post, label_type, label_pre, label_post) in enumerate(dataset, 1):
        with torch.no_grad():

            label_pre = label_pre.to(args.device)
            label_post = label_post.to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type, teacher_forcing_ratio=0)

            len_pre = label_pre.size(-1) 
            loss = (F.cross_entropy(pre_out.reshape(-1, len_pre), label_pre.argmax(-1).reshape(-1)) + \
                    F.cross_entropy(post_out.reshape(-1, len_pre), label_post.argmax(-1).reshape(-1)) ) / 2

            # log_prob_pre = F.log_softmax(pre_out, dim=-1) * label_pre
            # log_prob_post = F.log_softmax(post_out, dim=-1) * label_post
            # loss = -1 * (log_prob_pre.mean() + log_prob_post.mean())


            predicted_pre = F.one_hot(pre_out.argmax(-1), num_classes=pre_out.size(-1) ).reshape(-1)
            predicted_post = F.one_hot(post_out.argmax(-1), num_classes=post_out.size(-1)).reshape(-1)

            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)

            acc_point += (predicted_pre.eq(label_pre).sum().item() / predicted_pre.size(0) + 
                        predicted_pre.eq(label_pre).sum().item() / predicted_post.size(0) ) /2.

            precision_pre, recall_pre = torchmetrics.functional.precision_recall(predicted_pre, label_pre.int(), multiclass=False)
            precision_post, recall_post = torchmetrics.functional.precision_recall(predicted_post, label_post.int(), multiclass=False)
            valid_precision += (precision_pre+precision_post)/2.
            valid_recall += (recall_pre+recall_post)/2.
            
            valid_f1 += (torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False) \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)) / 2

            valid_loss += loss.data
            # print(predicted_pre, label_pre, total, correct, valid_f1)

    
    valid_loss /= i
    valid_acc = 100. * acc_point / i
    valid_f1 = valid_f1 / i
    valid_precision = valid_precision / i
    valid_recall = valid_recall / i
    return {'loss': valid_loss, 'acc_point': valid_acc, 'f1': valid_f1, 'precision': valid_precision, 'recall': valid_recall}

