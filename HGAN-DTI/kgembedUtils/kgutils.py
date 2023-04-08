from dgl import DGLGraph
import numpy as np
import torch
from time import time
import logging

def comp_deg_norm(g):
    np.seterr(divide='ignore', invalid='ignore')
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    logging.info('Single nodes = {}'.format((norm==0).sum()))
    return norm

def build_graph(args, net, idx=None):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors use reversed relations.
        This function also generates edge type and normalization factor (reciprocal of node incoming degree)

        Drug-0, Protein-1, Disease-2, Side-effect-3
    """
    num_nodes = args.nentity
    g = DGLGraph()
    g.add_nodes(num_nodes)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})

    i = 0
    #+++++++++++++++++++++++++
    # drug-drug triu
    drug_drug = net.drug_drug - np.diag(np.diag(net.drug_drug))  # remove self-loop
    drug_drug = np.triu(drug_drug)
    idx0 = np.nonzero(drug_drug)
    g.add_edges(idx0[0], idx0[1], {'e_label': i * torch.ones(len(idx0[0]), 1, dtype=torch.long)})
    i += 1

    # protein-protein triu
    protein_protein = net.protein_protein - np.diag(np.diag(net.protein_protein))  # remove self-loop
    protein_protein = np.triu(protein_protein)
    idx9 = np.nonzero(protein_protein)
    g.add_edges(idx9[0] + net.num_drug, idx9[1] + net.num_drug,
                {'e_label': i * torch.ones(len(idx9[0]), 1, dtype=torch.long)})
    i += 1

    # drug-protein
    if idx is not None:
        drug_protein = np.zeros_like(net.drug_protein)
        drug_protein[idx[0], idx[1]] = net.drug_protein[idx[0], idx[1]]
    else:
        drug_protein = net.drug_protein
    idx1 = np.nonzero(drug_protein)
    g.add_edges(idx1[0], idx1[1] + net.num_drug, {'e_label': i * torch.ones(len(idx1[0]), 1, dtype=torch.long)})
    i += 1

    # drug-disease
    drug_disease = net.drug_disease
    idx3 = np.nonzero(drug_disease)
    g.add_edges(idx3[0], idx3[1] + net.num_drug + net.num_protein,
                {'e_label': i * torch.ones(len(idx3[0]), 1, dtype=torch.long)})
    i += 1

    # drug_se
    drug_se = net.drug_sideEffect
    idx5 = np.nonzero(drug_se)
    g.add_edges(idx5[0], idx5[1] + net.num_drug + net.num_protein + net.num_disease,
                {'e_label': i * torch.ones(len(idx5[0]), 1, dtype=torch.long)})
    i += 1

    # protein_disease
    protein_disease = net.protein_disease
    idx7 = np.nonzero(protein_disease)
    g.add_edges(idx7[0] + net.num_drug, idx7[1] + net.num_drug + net.num_protein,
                {'e_label': i * torch.ones(len(idx7[0]), 1, dtype=torch.long)})
    i += 1

    num_one_dir_edges = g.number_of_edges()

    # drug-drug tril
    drug_drug_tril = drug_drug.T
    idx0_tril = np.nonzero(drug_drug_tril)
    g.add_edges(idx0_tril[0], idx0_tril[1], {'e_label': 0 * torch.ones(len(idx0_tril[0]), 1, dtype=torch.long)})

    # protein-protein tril
    protein_protein_tril = protein_protein.T
    idx9_tril = np.nonzero(protein_protein_tril)
    g.add_edges(idx9_tril[0] + net.num_drug, idx9_tril[1] + net.num_drug,
                {'e_label': 1 * torch.ones(len(idx9_tril[0]), 1, dtype=torch.long)})

    # protein-drug
    protein_drug = drug_protein.T
    idx2 = np.nonzero(protein_drug)
    g.add_edges(idx2[0] + net.num_drug, idx2[1], {'e_label': i * torch.ones(len(idx2[0]), 1, dtype=torch.long)})
    i += 1

    # disease-drug
    disease_drug = drug_disease.T
    idx4 = np.nonzero(disease_drug)
    g.add_edges(idx4[0] + net.num_drug + net.num_protein, idx4[1], {'e_label': i * torch.ones(len(idx4[0]), 1, dtype=torch.long)})
    i += 1

    # se_drug
    se_drug = drug_se.T
    idx6 = np.nonzero(se_drug)
    g.add_edges(idx6[0] + net.num_drug + net.num_protein + net.num_disease, idx6[1],
                {'e_label': i * torch.ones(len(idx6[0]), 1, dtype=torch.long)})
    i += 1

    # disease_protein
    disease_protein = protein_disease.T
    idx8 = np.nonzero(disease_protein)
    g.add_edges(idx8[0] + net.num_drug + net.num_protein, idx8[1] + net.num_drug,
                {'e_label': i * torch.ones(len(idx8[0]), 1, dtype=torch.long)})
    i += 1

    # add self-loop
    if args.self_loop == 1:
        g.add_edges(g.nodes(), g.nodes(), {'e_label': i * torch.ones(g.number_of_nodes(), 1, dtype=torch.long)})

    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata['e_id'] = edge_id

    return g, num_one_dir_edges, i + 1

def build_graph_zheng(args, net, idx=None):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors use reversed relations.
        This function also generates edge type and normalization factor (reciprocal of node incoming degree)

        Drug-0, Protein-1, drug substitutes-2, drug chemical structures-3, Side-effect-4, go term-5
    """
    num_nodes = args.nentity
    g = DGLGraph()
    g.add_nodes(num_nodes)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})

    i = 0
    #+++++++++++++++++++++++++

    # drug-protein
    if idx is not None:
        drug_protein = np.zeros_like(net.drug_protein)
        drug_protein[idx[0], idx[1]] = net.drug_protein[idx[0], idx[1]]
    else:
        drug_protein = net.drug_protein
    idx1 = np.nonzero(drug_protein)
    g.add_edges(idx1[0], idx1[1] + net.num_drug, {'e_label': i * torch.ones(len(idx1[0]), 1, dtype=torch.long)})
    i += 1

    # drug-chemical structure
    drug_st = net.drug_structure
    idx3 = np.nonzero(drug_st)
    g.add_edges(idx3[0], idx3[1] + net.num_drug + net.num_protein,
                {'e_label': i * torch.ones(len(idx3[0]), 1, dtype=torch.long)})
    i += 1

    # drug-substituent
    drug_su = net.drug_substituent
    idx4 = np.nonzero(drug_su)
    g.add_edges(idx4[0], idx4[1] + net.num_drug + net.num_protein + net.num_st,
                {'e_label': i * torch.ones(len(idx4[0]), 1, dtype=torch.long)})
    i += 1

    # drug_se
    drug_se = net.drug_sideEffect
    idx5 = np.nonzero(drug_se)
    g.add_edges(idx5[0], idx5[1] + net.num_drug + net.num_protein + net.num_st + net.num_su,
                {'e_label': i * torch.ones(len(idx5[0]), 1, dtype=torch.long)})
    i += 1

    # protein_go
    protein_go = net.protein_go
    idx7 = np.nonzero(protein_go)
    g.add_edges(idx7[0] + net.num_drug, idx7[1] + net.num_drug + net.num_protein + net.num_st + net.num_su + net.num_se,
                {'e_label': i * torch.ones(len(idx7[0]), 1, dtype=torch.long)})
    i += 1

    num_one_dir_edges = g.number_of_edges()

    # protein-drug
    protein_drug = drug_protein.T
    idx2 = np.nonzero(protein_drug)
    g.add_edges(idx2[0] + net.num_drug, idx2[1], {'e_label': i * torch.ones(len(idx2[0]), 1, dtype=torch.long)})
    i += 1

    # chemical structure-drug
    st_drug = net.drug_structure.T
    idx32 = np.nonzero(st_drug)
    g.add_edges(idx32[0] + net.num_drug + net.num_protein, idx32[1],
                {'e_label': i * torch.ones(len(idx32[0]), 1, dtype=torch.long)})
    i += 1

    # substituent-drug
    su_drug = net.drug_substituent.T
    idx42 = np.nonzero(su_drug)
    g.add_edges(idx42[0] + net.num_drug + net.num_protein + net.num_st, idx42[1],
                {'e_label': i * torch.ones(len(idx42[0]), 1, dtype=torch.long)})
    i += 1

    # se_drug
    se_drug = drug_se.T
    idx6 = np.nonzero(se_drug)
    g.add_edges(idx6[0] + net.num_drug + net.num_protein + net.num_st + net.num_su, idx6[1],
                {'e_label': i * torch.ones(len(idx6[0]), 1, dtype=torch.long)})
    i += 1

    # go-protein
    go_protein = net.protein_go
    idx72 = np.nonzero(go_protein)
    g.add_edges(idx72[0] + net.num_drug + net.num_protein + net.num_st + net.num_su + net.num_se, idx72[1] + net.num_drug,
                {'e_label': i * torch.ones(len(idx72[0]), 1, dtype=torch.long)})
    i += 1

    # add self-loop
    if args.self_loop == 1:
        g.add_edges(g.nodes(), g.nodes(), {'e_label': i * torch.ones(g.number_of_nodes(), 1, dtype=torch.long)})

    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata['e_id'] = edge_id

    return g, num_one_dir_edges, i + 1

def deep_dgl_graph_copy(graph: DGLGraph):
    start = time()
    copy_graph = DGLGraph()
    copy_graph.add_nodes(graph.number_of_nodes())
    graph_edges = graph.edges()
    copy_graph.add_edges(graph_edges[0], graph_edges[1])
    for key, value in graph.edata.items():
        copy_graph.edata[key] = value
    for key, value in graph.ndata.items():
        copy_graph.ndata[key] = value
    print('Graph copy take {:.2f} seconds'.format(time() - start))
    return copy_graph