from utils import *
from evaluation import evaluate_ranking
import random
import time
from model import Model


def training(dataLoader, config, device):
    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    if numUsers % config.batch_size == 0:
        numBatch = numUsers // config.batch_size
    else:
        numBatch = numUsers // config.batch_size + 1
    idxList = [i for i in range(numUsers)]

    model = Model(config, numItems).to(device)

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                        centered=False)

    for epoch in range(config.epochs):
        random.seed(1234)
        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batch_size * batch
            end = min(numUsers, start + config.batch_size)
            batchList = idxList[start:end]
            samples, uHis, target = generateBatchSamples(dataLoader, batchList, config, isEval=0)
            samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
            uHis = torch.from_numpy(uHis).type(torch.FloatTensor).to(device)
            target = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            scores = model.forward(samples, uHis, device)
            loss = -(torch.log(scores) * target + torch.log(1 - scores) * (1 - target)).sum(-1).mean()
            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd = time.time()

        if epoch % config.evalEpoch == 0:
            timeEvalStar = time.time()
            print("start evaluation")

            recall, ndcg = evaluate_ranking(model, dataLoader, config, device, config.isTrain)
            timeEvalEnd = time.time()
            output_str = "Epoch %d \t recall@5=%.8f, recall@10=%.8f, recall@20=%.8f," \
                         "ndcg@5=%.8f, ndcg@10=%.8f, ndcg@20=%.8f, [%.1f s]" % (
                             epoch + 1, recall[0], recall[1], recall[2], ndcg[0], ndcg[1], ndcg[2],
                             timeEvalEnd - timeEvalStar)
            print("time: %.1f, loss: %.3f" % (timeEpEnd - timeEpStr, epochLoss))
            print(output_str)
