import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn import metrics
import os
import copy

def train(
        model,
        x_train,  # 训练数据
        y_train,  # 训练标签
        x_test,
        y_test,
        n_epochs,  # 训练轮数
        batch_size,  # 批大小
        lr,  # 学习率
        verbose=True,
        save_model=True,
        root='./'
        ):
    train_dataset = Data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_dataset = Data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-7, eps=1e-08)

    min_loss_val = 10
    max_val_acc = 0.5
    # Training Model
    best_model = None
    loss_list = []
    valid_loss_list = []
    val_acc_list = []
    print('Model Training:')
    for epoch in range(n_epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        loss_ = 0
        for idx, (x, target) in enumerate(train_loader, 0):
            x = x.cuda()
            target = target.cuda()
            predict = model(x)
            #            losses.append(loss)
            correct += int(torch.sum(torch.argmax(predict, dim=1) == torch.argmax(target, dim=1)))
            total += len(target)
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            epoch_loss += loss.item()
            loss_ = epoch_loss / (idx + 1)
            del x, target, predict
            torch.cuda.empty_cache()
        loss_list.append(loss_)

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x_test).cuda())
            y = torch.tensor(y_test).cuda()
            val_cor = int(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)))
            val_acc = val_cor/len(y)
            val_acc_list.append(val_acc)
            valid_loss = criterion(pred, y)
            valid_loss_list.append(valid_loss.item())
            if val_acc >= max_val_acc and epoch + 1 >= 10 and save_model:
                max_val_acc = val_acc
                best_model = copy.deepcopy(model)
                torch.save(best_model, os.path.join(root, 'best_model.pth'))
        if verbose:
            print("Epoch={}/{}, loss={}, val_loss={}, lr={}, val_acc={}".format(
                epoch + 1, n_epochs, loss_, valid_loss, optimizer.state_dict()['param_groups'][0]['lr'], val_acc))
    print("Done")
    return best_model, loss_list, valid_loss_list, val_acc_list

def test_Model(test_data, test_label, model):
    model.eval()
    with torch.no_grad():
        test_data = torch.from_numpy(test_data).cuda()
        test_label = torch.from_numpy(test_label).cuda()
        outputs = model(test_data)
        Acc = 0
        Auc = 0
        Precision = 0
        F1 = 0
        Recall = 0
        predicted = torch.argmax(outputs, dim=1).cpu().numpy().astype('int64')
        labels = torch.argmax(test_label, dim=1).cpu().numpy().astype('int64')
        try:
            Acc = metrics.accuracy_score(labels, predicted)
            Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
            Auc = metrics.roc_auc_score(labels, predicted)
            F1 = metrics.f1_score(labels, predicted)
            Recall = metrics.recall_score(labels, predicted)
    #        print("label:",labels)
    #        print("prediction:", predicted)
            print('Acc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc, Auc, Precision, F1, Recall))
        except:
            pass
    return [Acc, Auc, Precision, F1, Recall], predicted