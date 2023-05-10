import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

class Model(nn.Module):
    def __init__(self, config, numItems):
        super(Model, self).__init__()

        # self
        self.config = config
        self.dim = self.config.dim
        self.asp = self.config.asp
        self.h = self.config.h
        self.nitems = numItems
        self.nbs = self.config.nbNUM  # neighbors

        self.itemEmb = nn.Embedding(numItems + 1, self.dim, padding_idx=config.padIdx)
        self.d_h = nn.Linear(self.dim, self.h)

        # Aspect-Specific Projection Matrices (K different aspects)
        self.aspProju = nn.Parameter(torch.Tensor(self.asp, 1, self.h), requires_grad=True)
        self.aspProji = nn.Parameter(torch.Tensor(self.asp, self.dim, self.h), requires_grad=True)
        torch.nn.init.xavier_normal_(self.aspProju.data, gain=1)
        torch.nn.init.xavier_normal_(self.aspProji.data, gain=1)
        self.out = nn.Linear(self.h * 2, numItems)
        self.his_embds = nn.Linear(numItems, self.h * 2)
        self.gate_trans1 = nn.Linear(self.h * 2, 1)
        self.gate_trans2 = nn.Linear(self.h * 2, 1)

        self.rnn = nn.LSTM(
            input_size=self.h * 2,
            hidden_size=self.h * 2,
            dropout=0.5,
            num_layers=2,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.rnn.flatten_parameters()
        self.dropout = nn.Dropout(0.5)

    def forward(self, seq, uHis, device):
        batch = seq.shape[0]  # batch
        self.max_seq = seq.shape[1]  # L
        self.max_bas = seq.shape[2]  # B

        embs = self.itemEmb(seq)  # [batch, L, B, d]
        user_embs = embs.view(batch, -1)  # [batch, L*B*d]

        # Aspect Layer
        userAsp = self.MIE_user(user_embs, device)  # [batch, asp, h]
        itemAsp = self.MIE_item(embs, device)  # [batch, asp, L, B, h]
        neighbAsps = self.SNA(userAsp, itemAsp, embs)  # [batch, asp, L, h*2]

        # Prediction
        NeiAsp4Max = neighbAsps.view(batch, self.max_seq, -1, self.asp)
        UaspMax = torch.max_pool2d(NeiAsp4Max, kernel_size=(1, self.asp)).squeeze(3)  # [batch, numItems]
        embs_max = UaspMax.view(batch, self.max_seq, -1)  # [batch, L, h*2]
        UaspAgg, (_, _) = self.rnn(embs_max, None)  # [batch, L, h*6]
        UaspAgg_h = torch.tanh(UaspAgg.sum(1))  # [batch, h*6]
        UaspAgg_n = self.out(UaspAgg_h)
        UaspAgg_n = F.softmax(UaspAgg_n, dim=-1)
        uHis_h = self.his_embds(uHis)  # [n, h*2]
        gate = torch.sigmoid(self.gate_trans1(uHis_h) + self.gate_trans2(UaspAgg_h))
        scores = gate * UaspAgg_n + (1 - gate) * uHis
        return scores

    def MIE_user(self, embs, device):
        '''
        input:
            embs [batch, L*B*d]
        output:
            userAsp  [batch, asp, h]
        '''

        self.fc1 = nn.Linear(embs.shape[1], 1, bias=True).to(device)  # [batch, L, B, d]
        u_embs = self.fc1(embs)  # [batch, 1]

        self.aspEmbed = nn.Embedding(self.asp, self.h).to(device)
        self.aspEmbed.weight.requires_grad = True
        torch.nn.init.xavier_normal_(self.aspEmbed.weight.data, gain=1)

        # Loop over all aspects
        userAsp_lst = []
        for a in range(self.asp):
            # [batch, 1] × [1, h] = [batch, h]
            user_aspProj = torch.tanh(torch.matmul(u_embs, self.aspProju[a]))  # [asp, h]
            userAsp_lst.append(torch.unsqueeze(user_aspProj, 1))  # [batch, 1, h]

        # [batch, asp, h]
        userAsp = torch.cat(userAsp_lst, dim=1)

        return userAsp

    def MIE_item(self, embs, device):
        '''
        input:
            embs [batch,L,B,d]
        output:
            itemAsp  [batch, asp, L, B, h]
        '''

        self.aspEmbed = nn.Embedding(self.asp, self.h).to(device)
        self.aspEmbed.weight.requires_grad = True
        torch.nn.init.xavier_normal_(self.aspEmbed.weight.data, gain=1)

        # Loop over all aspects
        itemAsp_lst = []
        for a in range(self.asp):
            item_aspProj = torch.tanh(torch.matmul(embs, self.aspProji[a]))  # [batch, L, B, h]
            itemAsp_lst.append(torch.unsqueeze(item_aspProj, 1))  # [batch,1,L,B,h]

        # [batch,asp,L,B,h]
        itemAsp = torch.cat(itemAsp_lst, dim=1)

        return itemAsp  # [batch,asp,L,B,h]

    def SNA(self, userAsp, itemAsp, embs):
        NeiAsp_lst = list()
        # Over loop each aspect
        for i in range(userAsp.shape[1]):
            userAsp_sim = torch.tanh(userAsp[:, i, :])  # [batch, h]
            embs_asp = torch.tanh(itemAsp[:, i, :])  # [batch, L, B, h]

            # 相似度计算(userAsp_sim)
            userAsp_sim_np = userAsp_sim.cpu().detach().numpy().astype('float32')  # [batch, L, B, d]
            # faiss
            index = faiss.IndexFlatL2(self.h)  # build the index
            index.add(userAsp_sim_np)
            k = self.nbs  # neighbors
            _, I = index.search(userAsp_sim_np, k)  # (distance, ID)

            if embs.shape[0] < 100:
                nb_list = list()
                for i in range(embs.shape[0]):
                    nb = embs[I[i, 1:]].unsqueeze(0)  # [batch, k-1, L, B, d]
                    nb_list.append(nb)
                neighb = torch.cat(nb_list, dim=0)  # [batch, k-1, L, B, d]
            else:
                neighb = embs[I[:, 1:]]  # [batch, k-1, L, B, d]
            neighb_sum = self.d_h(torch.sum(neighb, dim=1))  # [batch, L, B, h]
            embs_UI = torch.sum(embs_asp, dim=2)  # [batch, L, h]
            embs_PI = torch.sum(neighb_sum, dim=2)  # [batch, L, h]
            nb_embs = torch.cat([embs_UI, embs_PI], dim=2)  # [batch, L, h*2]
            NeiAsp_lst.append(nb_embs.unsqueeze(1))  # [batch, 1, L, h*2]
        neighbAsps = torch.cat(NeiAsp_lst, dim=1)  # [batch, asp, L, h*2]
        return neighbAsps  # [batch, asp, L, h*2]
