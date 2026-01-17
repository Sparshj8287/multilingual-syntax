"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(L1DepthLoss, self).__init__()
    self.args = args
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class CrossEntropyLoss(nn.Module):

  def __init__(self, args):
    super(CrossEntropyLoss, self).__init__()
    print('Constructing CrossEntropyLoss')
    self.args = args
    self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

  def forward(self, predictions, label_batch, length_batch):
    """
    Computes and returns CrossEntropyLoss.
    """
    if len(label_batch.size()) == 2:
      batchlen, seqlen, class_count = predictions.size()
      total_sents = torch.sum((length_batch != 0)).float()
      predictions = predictions.view(batchlen*seqlen, class_count)
      label_batch = label_batch.view(batchlen*seqlen).long()
      cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch) / total_sents
    #else:
    #  batchlen, seqlen, seqlen = predictions.size()
    #  total_sents = torch.sum((length_batch != 0)).float()
    #  predictions = predictions.view(batchlen*seqlen*seqlen)
    #  label_batch = label_batch.view(batchlen*seqlen*seqlen).float()
    #  cross_entropy_loss = self.pytorch_bce_loss(predictions, label_batch)
    return cross_entropy_loss, total_sents


class PolarProbeLoss(nn.Module):
  """Joint distance + angular loss for Polar Probe."""

  def __init__(self, args):
    super(PolarProbeLoss, self).__init__()
    self.args = args
    self.distance_loss = L1DistanceLoss(args)
    self.angular_lambda = args['probe_training'].get('angular_lambda',
        args.get('probe', {}).get('angular_lambda', 0.5))
    self.mse_loss = nn.MSELoss()

  def angular_loss(self, projected_batch, observation_batch, length_batch):
    edges = []
    rels = []
    device = projected_batch.device

    for sent_index, (observation, _) in enumerate(observation_batch):
      length = int(length_batch[sent_index].item())
      head_indices = observation.head_indices[:length]
      rel_labels = observation.governance_relations[:length] if observation.governance_relations else []
      for dep_index, head in enumerate(head_indices):
        if head in (None, '_'):
          continue
        try:
          head_index = int(head)
        except (TypeError, ValueError):
          continue
        if head_index == 0 or head_index > length:
          continue
        rel = rel_labels[dep_index] if rel_labels else None
        if rel in (None, '_'):
          continue
        rel = rel.split(':')[0]
        edge = projected_batch[sent_index, head_index - 1] - projected_batch[sent_index, dep_index]
        edges.append(edge)
        rels.append(rel)

    if len(edges) < 2:
      return torch.tensor(0.0, device=device)

    edges = torch.stack(edges)
    unique_rels = sorted(set(rels))
    rel_to_idx = {rel: i for i, rel in enumerate(unique_rels)}
    rels_tensor = torch.tensor([rel_to_idx[rel] for rel in rels], device=device)

    sorted_indices = torch.argsort(rels_tensor)
    rels_tensor = rels_tensor[sorted_indices]
    edges = edges[sorted_indices]

    norm_edges = F.normalize(edges, p=2, dim=1)
    cos_sim_mat = torch.mm(norm_edges, norm_edges.t())
    cos_dist_mat = 1 - cos_sim_mat
    cos_dist_mat = torch.clamp(cos_dist_mat, min=0.0, max=2.0)

    gold_mat = torch.ones_like(cos_dist_mat)
    for rel in torch.unique(rels_tensor):
      indices = (rels_tensor == rel).nonzero(as_tuple=True)[0]
      gold_mat[indices[:, None], indices] = 0

    return self.mse_loss(cos_dist_mat, gold_mat)

  def forward(self, predictions, label_batch, length_batch, observation_batch, projected_batch):
    dist_loss, total_sents = self.distance_loss(predictions, label_batch, length_batch)
    ang_loss = self.angular_loss(projected_batch, observation_batch, length_batch)
    total_loss = dist_loss + (self.angular_lambda * ang_loss)
    return total_loss, total_sents
