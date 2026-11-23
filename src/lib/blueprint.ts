export interface ConnectedBlock { id: string; inputs: string[] }

export function isSequence(blocks: ConnectedBlock[]): boolean {
  return blocks.every((block, index) => index === 0
    ? block.inputs.length === 0
    : block.inputs.length === 1 && block.inputs[0] === blocks[index - 1].id)
}

export function connectSequence<T extends ConnectedBlock>(blocks: T[]): T[] {
  return blocks.map((block, index) => ({ ...block, inputs: index === 0 ? [] : [blocks[index - 1].id] }))
}

export function syncBlueprintDimensions<T extends ConnectedBlock & { type: string; params: Record<string, unknown> }>(
  blocks: T[], before: { hidden: number; vocab: number; positions: number },
  after: { hidden: number; vocab: number; positions: number },
): T[] {
  return blocks.map(block => {
    const params = { ...block.params }
    if (['token_embedding', 'positional_embedding'].includes(block.type) && params.hidden_size === before.hidden)
      params.hidden_size = after.hidden
    if (block.type === 'token_embedding' && params.vocab_size === before.vocab) params.vocab_size = after.vocab
    if (block.type === 'positional_embedding' && params.max_position_embeddings === before.positions)
      params.max_position_embeddings = after.positions
    if (params.intermediate_size === before.hidden * 4) params.intermediate_size = after.hidden * 4
    for (const key of ['in_features', 'out_features'])
      if (params[key] === before.hidden) params[key] = after.hidden
    return { ...block, params }
  })
}
