import type { ParamInfoEntry } from './types'

// UI-only overrides (do not affect backend keys or payloads)
export const PARAM_LABEL_OVERRIDES: Record<string, string> = {
  nu: "Poisson's Ratio (ν)",
  Emod: 'Elastic Modulus (E)',
  Kx: 'Soil Stiffness (Horizontal)',
  Ky: 'Soil Stiffness (Transverse)',
  Kz: 'Soil Stiffness (Vertical)',
}

export function getParamDisplayLabel(key: string, meta?: ParamInfoEntry | null): string {
  const fallback =
    (typeof meta?.name === 'string' && meta.name) ||
    (typeof (meta as any)?.technical_name === 'string' && (meta as any).technical_name) ||
    key

  return PARAM_LABEL_OVERRIDES[key] ?? fallback
}

const ASSISTANT_LABEL_PHRASE_OVERRIDES: Array<[from: string, to: string]> = [
  ['Concrete Flexibility Ratio', "Poisson's Ratio (ν)"],
  ['Concrete Stiffness', 'Elastic Modulus (E)'],
  ['Ground Support (Horizontal)', 'Soil Stiffness (Horizontal)'],
  ['Ground Support (Sideways)', 'Soil Stiffness (Transverse)'],
  ['Ground Support (Vertical)', 'Soil Stiffness (Vertical)'],
]

// Display-time transformation for assistant messages only.
// Keep it narrowly scoped to specific phrases to avoid altering meaning.
export function transformAssistantDisplayText(text: string): string {
  if (!text) return text
  let out = text
  for (const [from, to] of ASSISTANT_LABEL_PHRASE_OVERRIDES) {
    out = out.split(from).join(to)
  }
  return out
}
