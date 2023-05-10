import numpy as np
import torch


def generateBatchSamples(dataLoader, batchIdx, config, isEval):
    samples, sampleLen, uHis, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)
    maxLenSeq = max([len(userLen) for userLen in sampleLen])  # max length of sequence
    maxLenBas = max([max(userLen) for userLen in sampleLen])  # max length of basket

    # pad users
    paddedSamples = []
    lenList = []
    for user in samples:
        trainU = user[:-1]
        paddedU = []
        lenList.append([len(trainU) - 1])
        for eachBas in trainU:
            paddedBas = eachBas + [config.padIdx] * (maxLenBas - len(eachBas))
            paddedU.append(paddedBas)  # [batch, maxLenBas]
        paddedU = paddedU + [[config.padIdx] * maxLenBas] * (maxLenSeq - len(paddedU))
        # add a sample
        paddedSamples.append(paddedU)  # [batch, maxLenSeq]

    # 1-hot vectors
    lenTen = torch.tensor(lenList, dtype=torch.long)  # [batch,1]
    lenX = torch.FloatTensor(len(samples), maxLenSeq)  # [batch, maxLenSeq]
    lenX.zero_()
    lenX.scatter_(1, lenTen, 1)  # [batch, maxLenSeq]
    return np.asarray(paddedSamples), uHis, target
