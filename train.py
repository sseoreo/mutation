import os
import math

from sklearn import multiclass
import torch
import torch.nn.functional as F

import torchmetrics.functional 

def train(args, model, optimizer, trainset, validset, logger=None):
    if args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type_attn']:
        train_type(args, model, optimizer, trainset, validset, logger)

    elif args.mode in ['single_point', 'single_point_attn', 'seq2seq_point', 'seq2seq_point_attn']:
        train_point(args, model, optimizer, trainset, validset, logger)

    elif args.mode in ['single_all', 'single_all_attn', 'seq2seq_all', 'seq2seq_all_attn']:
        train_all(args, model, optimizer, trainset, validset, logger)

    else:
        raise Exception(f"Not defined mode: {args.mode}")



def evaluate(args, model, dataset):
    if args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type', 'seq2seq_type_attn']:
        return eval_type(args, model, dataset)
    
    elif args.mode in ['single_point', 'single_point']:
        return eval_point(args, model, dataset)
    
    elif args.mode in ['single_all', 'single_all_attn', 'seq2seq_all', 'seq2seq_all_attn']:
        return eval_all(args, model, optimizer, trainset, validset, logger)

    else:
        raise Exception(f"Not defined mode: {args.mode}")


def train_type(args, model, optimizer, trainset, validset, logger):
    
    best_acc = 0.
    for epoch in range(args.epochs):
        
        print(f"[{epoch}/{args.epochs}] epochs training...")
        train_loss, total, correct = 0., 0, 0
        model.train()        
        for i, (pre, post, label_type, _, _) in enumerate(trainset, 1):
            
            label_type = label_type.to(args.device)
            out = model(pre.to(args.device), post.to(args.device), label_type)
            
            label = label_type.reshape(-1)
            
            loss = F.cross_entropy(out.reshape(-1, args.vocab_size), label, reduction='sum')

            predicted = out.argmax(-1).reshape(-1)


            total += label.size(0)
            # print(predicted.shape, label.shape, predicted.eq(label))
            correct += predicted.eq(label).sum().item()
            train_loss += loss.data

            # print(correct/total, correct, total)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
        
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / total,
                # 'train/ppl': math.exp(train_loss / (i+1)),
                'train/acc': 100. * correct / total 
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
                    'valid/acc': valid_acc 
                })    
        print(f'Epoch : {epoch} Done!')


def eval_type(args, model, dataset):
    model.eval()
            
    valid_loss, total, correct = 0., 0, 0
    for i, (pre, post, label_type, _, _) in enumerate(dataset, 0):
        with torch.no_grad():

            label_type = label_type.to(args.device)
            out = model(pre.to(args.device), post.to(args.device), label_type)

            label = label_type.reshape(-1)
            loss = F.cross_entropy(out.reshape(-1, args.vocab_size), label, reduction='sum')

            predicted = out.argmax(-1).reshape(-1)


            total += label.size(0)
            # print(predicted.shape, label.shape, predicted.eq(label))
            correct += predicted.eq(label).sum().item()
            valid_loss += loss.data
    
    valid_loss /= total
    valid_acc = 100. * correct / total
    return valid_loss, valid_acc



def train_point(args, model, optimizer, trainset, validset, logger):
    
    train_f1 = 0.
    best_f1 = 0.
    for epoch in range(args.epochs):
        
        print(f"[{epoch}/{args.epochs}] epochs training...")
        train_loss, total, correct = 0., 0, 0
        model.train()        
        for i, (pre, post, label_type, label_pre, label_post) in enumerate(trainset, 1):
            
            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            # print(pre_out.shape, post_out.shape, label_pre.shape)

            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)
            loss = F.binary_cross_entropy(pre_out.reshape(-1), label_pre, reduction='sum') + \
                        F.binary_cross_entropy(post_out.reshape(-1), label_post, reduction='sum')

            predicted_pre = (pre_out>0.5).int().reshape(-1)
            predicted_post = (post_out>0.5).int().reshape(-1)


            # print(predicted_pre.shape, label_pre.shape, predicted_pre == label_pre)

            total += predicted_pre.size(0) + predicted_post.size(0)
            correct += ( predicted_pre == label_pre).sum() + \
                                ( predicted_post == label_post).sum()
            train_f1 += torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False)*args.src_len \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)*args.src_len

            train_loss += loss.data

            # print(correct/total, correct, total)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
        
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / total,
                'train/f1': train_f1 / total,
                'train/acc': 100. * correct / total 
            })
        
        if epoch % args.valid_interval == 0:
            valid_loss, valid_acc, valid_f1 = eval_point(args, model, validset)
            print('[epoch %d/%d] val_loss: %.3f' %
                (epoch, args.epochs, valid_loss))            
            print('[epoch %d/%d] val_acc: %.3f' %
                (epoch, args.epochs, valid_acc))  
            print('[epoch %d/%d] val_f1: %.3f' %
                (epoch, args.epochs, valid_f1))           

                
            if best_f1 < valid_f1:
                best_f1 = valid_f1
                torch.save(model.state_dict(), 
                                    os.path.join(args.output_dir, 'best.pt') )
            
            if logger is not None:
                logger.log({
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'valid/loss': valid_loss,
                    'valid/f1': valid_f1,
                    # 'train/ppl': math.exp(train_loss / (i+1)),
                    'valid/acc': valid_acc 
                })    
        print(f'Epoch : {epoch} Done!')


def eval_point(args, model, dataset):
    model.eval()
            
    valid_loss, valid_f1, total, correct = 0., 0., 0, 0
    for i, (pre, post, label_type, label_pre, label_post) in enumerate(dataset, 0):
        with torch.no_grad():

            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            # pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_pre, label_post)
            # print(pre_out.shape, post_out.shape, label_pre.shape)

            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)

            loss = F.binary_cross_entropy(pre_out.reshape(-1), label_pre, reduction='sum') + \
                        F.binary_cross_entropy(post_out.reshape(-1), label_post, reduction='sum')

            predicted_pre = (pre_out>0.5).int().reshape(-1)
            predicted_post = (post_out>0.5).int().reshape(-1)


            total += predicted_pre.size(0) + predicted_post.size(0)
            correct += ( predicted_pre == label_pre).sum() + \
                                ( predicted_post == label_post).sum()
            valid_f1 += torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False)*args.src_len \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)*args.src_len

            # print(predicted_pre, label_pre, total, correct, valid_f1)

            valid_loss += loss.data
    
    valid_loss /= total
    valid_acc = 100. * correct / total
    valid_f1 = valid_f1 / total
    return valid_loss, valid_acc, valid_f1


def train_all(args, model, optimizer, trainset, validset, logger):
    
    best_f1 = 0.
    for epoch in range(args.epochs):
        
        print(f"[{epoch}/{args.epochs}] epochs training...")
        train_loss, train_f1 = 0., 0.
        acc_type, acc_point = 0, 0

        model.train()        
        for i, (pre, post, label_type, label_pre, label_post) in enumerate(trainset, 1):
            
            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)
            label_type = label_type.to(args.device)
            
            out_type, pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            # print(pre_out.shape, post_out.shape, label_pre.shape)

            label_type = label_type.reshape(-1)
            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)
            loss = F.binary_cross_entropy(pre_out.reshape(-1), label_pre, reduction='sum') + \
                        F.binary_cross_entropy(post_out.reshape(-1), label_post, reduction='sum')
            loss += F.cross_entropy(out_type.reshape(-1, args.vocab_size), label_type, reduction='sum')

            # print(correct/total, correct, total)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            

            predicted_pre = (pre_out>0.5).int().reshape(-1)
            predicted_post = (post_out>0.5).int().reshape(-1)
            predicted_type = out_type.argmax(-1).reshape(-1)


            # print(predicted_pre.shape, label_pre.shape, predicted_pre == label_pre)
            acc_type += predicted_type.eq(label_type).sum().item() / label_type.size(0)
            acc_point += ( (predicted_pre == label_pre).sum().item()+( predicted_post == label_post).sum().item() ) / predicted_pre.size(0) + predicted_post.size(0)
            train_f1 += (torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False) \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)) / 2

            train_loss += loss.data
            
            break
        
        if logger is not None:
            logger.log({
                'epoch': epoch,
                'train/loss': train_loss / i,
                'train/f1': train_f1 / i,
                'train/acc_type': 100. * acc_type / i,
                'train/acc_point': 100. * acc_point / i 

            })
        
        if epoch % args.valid_interval == 0:
            valid_loss, valid_acc, valid_f1 = eval_all(args, model, validset)
            print('[epoch %d/%d] val_loss: %.3f' %
                (epoch, args.epochs, valid_loss))            
            print('[epoch %d/%d] val_acc: %.3f' %
                (epoch, args.epochs, valid_acc))  
            print('[epoch %d/%d] val_f1: %.3f' %
                (epoch, args.epochs, valid_f1))           

                
            if best_f1 < valid_f1:
                best_f1 = valid_f1
                torch.save(model.state_dict(), 
                                    os.path.join(args.output_dir, 'best.pt') )
            
            if logger is not None:
                logger.log({
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'valid/loss': valid_loss,
                    'valid/f1': valid_f1,
                    # 'train/ppl': math.exp(train_loss / (i+1)),
                    'valid/acc_type': valid_acc[0] ,
                    'valid/acc_point': valid_acc[1]
                })    
        print(f'Epoch : {epoch} Done!')


def eval_all(args, model, dataset):
    model.eval()
            
    valid_loss, valid_f1, acc_type, acc_point = 0., 0., 0, 0
    for i, (pre, post, label_type, label_pre, label_post) in enumerate(dataset, 0):
        with torch.no_grad():

            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)
            label_type = label_type.to(args.device)
            
            out_type, pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_type)
            # print(pre_out.shape, post_out.shape, label_pre.shape)

            label_type = label_type.reshape(-1)
            label_pre = label_pre.reshape(-1)
            label_post = label_post.reshape(-1)
            loss = F.binary_cross_entropy(pre_out.reshape(-1), label_pre, reduction='sum') + \
                        F.binary_cross_entropy(post_out.reshape(-1), label_post, reduction='sum')
            loss += F.cross_entropy(out_type.reshape(-1, args.vocab_size), label_type, reduction='sum')

            

            predicted_pre = (pre_out>0.5).int().reshape(-1)
            predicted_post = (post_out>0.5).int().reshape(-1)
            predicted_type = out_type.argmax(-1).reshape(-1)


            # print(predicted_pre.shape, label_pre.shape, predicted_pre == label_pre)
            acc_type += predicted_type.eq(label_type).sum().item() / label_type.size(0)
            acc_point += ( (predicted_pre == label_pre).sum().item()+( predicted_post == label_post).sum().item() ) / predicted_pre.size(0) + predicted_post.size(0)
            valid_f1 += (torchmetrics.functional.f1_score(predicted_pre, label_pre.int(), multiclass=False) \
                            + torchmetrics.functional.f1_score(predicted_post, label_post.int(), multiclass=False)) / 2

            valid_loss += loss.data
    
    valid_loss /= i
    valid_acc_type = 100. * acc_type / i
    valid_acc_point = 100. * acc_point / i
    valid_f1 = valid_f1 / i
    return valid_loss, (valid_acc_type, valid_acc_point), valid_f1
