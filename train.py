import os
import math

from sklearn import multiclass
import torch
import torch.nn.functional as F

import torchmetrics.functional 

def train(args, model, optimizer, trainset, validset, logger=None):
    if args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type']:
        train_type(args, model, optimizer, trainset, validset, logger)
    elif args.mode in ['single_point', 'single_point_attn']:
        train_point(args, model, optimizer, trainset, validset, logger)
    else:
        raise Exception(f"Not defined mode: {args.mode}")



def evaluate(args, model, dataset):
    if args.mode in ['single_type', 'single_type_attn', 'seq2seq_type', 'seq2seq_type']:
        return eval_type(args, model, dataset)
    elif args.mode in ['single_point', 'single_point']:
        return eval_point(args, model, dataset)
    # elif args.mode in ['single_token', 'single_token_attn']:
    #     return eval_type_token(args, model, dataset, criterion)
    # elif args.mode in ['seq2seq', 'seq2seq_attn', 'seq2seq_point']:
    #     return eval_seq2seq(args, model, dataset, criterion)


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
        for i, (pre, post, _, label_pre, label_post) in enumerate(trainset, 1):
            
            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_pre, label_post)
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
    for i, (pre, post, _, label_pre, label_post) in enumerate(dataset, 0):
        with torch.no_grad():

            label_pre = label_pre.float().to(args.device)
            label_post = label_post.float().to(args.device)

            pre_out, post_out = model(pre.to(args.device), post.to(args.device), label_pre, label_post)
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


# def train_single_token(args, model, optimizer, criterion, trainset, validset, run):

#     best_acc = 0.
#     for epoch in range(args.epochs):
        
#         print(f"[{epoch}/{args.epochs}] epochs training...")
#         train_loss,  total, correct, train_f1 = 0., 0, 0, 0.
#         model.train()        
#         for i, (pre, post, pre_label, post_label) in enumerate(trainset, 1):
            
#             bsz, pre_len = pre.shape
#             _, post_len = post.shape

#             out = model(pre.cuda(), post.cuda()) # batch * 100 * 1
#             # print(out )
#             # out = torch.clip(out, min=0.1)
#             # print(out[:, :pre_len].shape, pre_label.shape)
#             loss = criterion(out[:,:pre_len,:].reshape(-1, pre_len), pre_label.cuda())
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
#             optimizer.step()

#             pre_label = pre_label.int().cuda()
#             out = (out[:,:pre_len,:].reshape(-1, pre_len)>0.5).int()
#             # out = (out[:,:pre_len]>0.5).int()
            
#             # correct += (out.reshape(args.batch_size,100) == torch.cat((pre_label.cuda(), post_label.cuda()), dim = 1)).float().sum()
#             correct += ( out == pre_label).float().sum()
#             train_f1 += torchmetrics.functional.f1_score(out, pre_label, multiclass=False)
            
#             total += pre_label.size(0) * pre_len # batch_size * 100

#             train_loss += loss.item()
            
        
#         if run is not None:
#             run['train/loss'].log(train_loss / (i+1))
#             run['train/f1'].log(train_f1/ (i+1))
#             run['train/acc'].log(100. * correct / total)
        
#         if epoch % args.valid_interval == 0:
#             valid_loss, valid_acc, valid_f1 = eval_single_token(args, model, validset, criterion)
#             print('[%d, %5d] val_loss: %.3f' %
#                 (epoch + 1, i + 1, valid_loss))            
#             print('[%d, %5d] val_acc: %.3f' %
#                 (epoch + 1, i + 1, valid_acc))
#             print('[%d, %5d] val_f1: %.3f' %
#                 (epoch + 1, i + 1, valid_f1))

#             if run is not None:
#                 run['valid/loss'].log(valid_loss)
#                 run['valid/f1'].log(valid_f1)
#                 run['valid/acc'].log(valid_acc)

#             if best_acc < valid_acc:
#                 best_acc = valid_acc
#                 torch.save(model.state_dict(), 
#                                     os.path.join(args.output_dir, 'best.pt') )
                
            
#         print(f'Epoch : {epoch} Done!')


# def eval_single_token(args, model, dataset, criterion):
#     model.eval()
            
#     valid_loss, total, correct, valid_f1 = 0., 0, 0, 0.
#     for i, (pre, post, pre_label, post_label) in enumerate(dataset, 0):
#         with torch.no_grad():
            
#             bsz, pre_len = pre.shape
#             _, post_len = post.shape

#             out = model(pre.cuda(), post.cuda()) # batch * 100 * 1
            
#             # loss = criterion(out[:, :pre_len], pre_label.cuda())
#             # loss = criterion(out.reshape(-1, pre_len+post_len), torch.cat((pre_label.cuda(), post_label.cuda()), dim = 1))
#             loss = criterion(out[:,:pre_len,:].reshape(-1, pre_len), pre_label.cuda())

            
#             pre_label = pre_label.int().cuda()
#             out = (out[:,:pre_len,:].reshape(-1, pre_len)>0.5).int()
#             # out = (out[:,:pre_len]>0.5).int()

#             # correct += (out.reshape(args.batch_size,100) == torch.cat((pre_label.cuda(), post_label.cuda()), dim = 1)).float().sum()
#             correct += ( out == pre_label).float().sum()
#             total += pre_label.size(0) * pre_len 
#             valid_f1 += torchmetrics.functional.f1_score(out, pre_label, multiclass=False)
            
#             valid_loss += loss.item()
    
#     valid_loss /= i+1
#     valid_acc = 100. * correct / total
#     valid_f1 = valid_f1 / (i+1)
#     return valid_loss, valid_acc, valid_f1
