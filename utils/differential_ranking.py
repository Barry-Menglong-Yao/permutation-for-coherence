def ranks(inputs, axis=-1):
  """Returns the ranks of the input values among the given axis."""
  return torch.argsort(torch.argsort(inputs, descending=False, dim=1), dim=1)+1