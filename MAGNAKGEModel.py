#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import numpy as np
import math
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from MAGNA_KGE.MAGNA_KGEncoder import MAGNAKGEncoder
from kge_losses.lossfunction import KGESmoothCELoss

class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self._nentity = args.nentity
        self._nrelation = args.nrelation
        self._nedges = args.nedges
        self._ent_emb_size = args.ent_embed_dim
        self._rel_emb_size = args.rel_embed_dim
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.entity_embedding = nn.Parameter(torch.zeros(args.nentity, self._ent_emb_size), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.zeros(args.nrelation, self._rel_emb_size), requires_grad=True)
        self.inp_drop = nn.Dropout(p=args.input_drop)
        self.feature_drop = nn.Dropout(p=args.fea_drop)
        self.graph_on = args.graph_on == 1
        self.project_on = args.project_on == 1
        self.args = args

        if (not self.graph_on) and (self._ent_emb_size != self._rel_emb_size):
            self.project_on = True

        if self.project_on:
            self.ent_map = nn.Linear(self._ent_emb_size, self.embed_dim, bias=False)
            self.rel_map = nn.Linear(self._rel_emb_size, self.embed_dim, bias=False)
            graph_ent_in_dim = self.embed_dim
            graph_rel_in_dim = self.embed_dim
        else:
            self.ent_map, self.rel_map = Identity(), Identity()
            graph_ent_in_dim = self._ent_emb_size
            graph_rel_in_dim = self._rel_emb_size

        self.dag_entity_encoder = MAGNAKGEncoder(
            in_ent_dim=graph_ent_in_dim,
            in_rel_dim=graph_rel_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            input_drop=args.input_drop,
            num_heads=args.num_heads,
            hop_num=args.hops,
            attn_drop=args.att_drop,
            feat_drop=args.fea_drop,
            negative_slope=args.slope,
            edge_drop=args.edge_drop,
            topk_type=args.topk_type,
            alpha=args.alpha,
            topk=args.top_k,
            ntriples=args.nedges,
            args=args)

        # if self.args.layer_feat_fusion == 'concatenation':
        #     self.fc = nn.Linear(args.hidden_dim * args.layers, args.hidden_dim)

        if self.args.layer_feat_fusion == 'concatenation':
            in_dim = args.hidden_dim * 2 * args.layers
        else:
            in_dim = args.hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.init()

    def init(self):
        sqrt_size = 6.0 / math.sqrt(self._ent_emb_size)
        nn.init.uniform_(tensor=self.entity_embedding, a=-sqrt_size, b=sqrt_size)
        sqrt_size = 6.0 / math.sqrt(self._rel_emb_size)
        nn.init.uniform_(tensor=self.entity_embedding, a=-sqrt_size, b=sqrt_size)
        if self.project_on and isinstance(self.rel_map, nn.Linear):
            nn.init.xavier_normal_(tensor=self.rel_map.weight.data, gain=1.414)
        if self.project_on and isinstance(self.ent_map, nn.Linear):
            nn.init.xavier_normal_(tensor=self.ent_map.weight.data, gain=1.414)

    def forward(self, graph, type_mask, index):
        graph = graph.local_var()

        entity_embedder, relation_embedder = self.ent_map(self.entity_embedding), self.rel_map(self.relation_embedding)
        if self.graph_on:
            entity_embedder = self.dag_entity_encoder(graph, entity_embedder, relation_embedder)

        if self.args.layer_feat_fusion == 'last':
            entity_embedder = entity_embedder[-1]  # only obtain the last later output
        elif self.args.layer_feat_fusion == 'average':
            entity_embedder = torch.mean(torch.stack(entity_embedder, dim=0), dim=0)
        elif self.args.layer_feat_fusion == 'max':
            entity_embedder, _ = torch.max(torch.stack(entity_embedder, dim=0), dim=0)
        elif self.args.layer_feat_fusion == 'concatenation':
            entity_embedder = torch.cat(entity_embedder, dim=-1)
            # entity_embedder = F.relu(self.fc(entity_embedder))

        drug_embed = entity_embedder[type_mask == 0]
        protein_embed = entity_embedder[type_mask == 1]
        comb_embed = torch.cat((drug_embed[index[0]], protein_embed[index[1]]), dim=1)
        output = self.classifier(comb_embed)
        logits = F.softmax(output, dim=-1)
        return logits

    def kge_loss_computation(self, sample, entity_embed=None, relation_embed=None):
        head_part, rel_part, tail_part = sample[:, 0], sample[:, 1], sample[:, 2]
        if entity_embed is None:
            entity_embedding = self.entity_embedding
        else:
            entity_embedding = entity_embed
        if relation_embed is None:
            relation_embedding = self.relation_embedding
        else:
            relation_embedding = relation_embed

        head_embed = torch.index_select(entity_embedding, dim=0, index=head_part)
        tail_embed = torch.index_select(entity_embedding, dim=0, index=tail_part)
        regularization = self.regularization * self.kge_loss_function(head_embed, tail_embed, relation_embedding, rel_part)
        return regularization
