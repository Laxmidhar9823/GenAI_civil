export type EngineeringValidationResult =
  | { ok: true; normalizedValue: string; warning?: string }
  | { ok: false; normalizedValue: string; error: string }

function toFiniteNumber(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}

function clampLowered(raw: string) {
  return raw.trim().toLowerCase()
}

function isFinitePositive(n: number | null): n is number {
  return n !== null && Number.isFinite(n) && n > 0
}

function isFiniteNonNegative(n: number | null): n is number {
  return n !== null && Number.isFinite(n) && n >= 0
}

export function validateInlineEngineeringEdit(args: {
  key: string
  rawValue: string
  currentParams: Record<string, unknown>
}): EngineeringValidationResult {
  const key = args.key.trim()
  const raw = args.rawValue
  const normalizedValue = raw.trim()

  if (!key) {
    return { ok: false, normalizedValue, error: 'Choose a field to edit.' }
  }

  if (!normalizedValue) {
    return { ok: false, normalizedValue, error: 'Enter a value.' }
  }

  // Derived node arrays are not editable.
  if (key === 'x' || key === 'y') {
    return { ok: false, normalizedValue, error: 'Node coordinates are calculated and cannot be edited.' }
  }

  const current = args.currentParams ?? {}

  // Tentative params snapshot with the proposed edit.
  const tentative: Record<string, unknown> = { ...current }

  // Enum keys
  if (key === 'mesh_type') {
    const v = clampLowered(normalizedValue)
    if (!['coarse', 'medium', 'fine'].includes(v)) {
      return { ok: false, normalizedValue, error: 'Mesh Density must be one of: coarse, medium, fine.' }
    }
    tentative.mesh_type = v
  } else if (key === 'load_location') {
    const v = clampLowered(normalizedValue)
    if (!['corner', 'edge', 'interior'].includes(v)) {
      return { ok: false, normalizedValue, error: 'Load Location must be one of: corner, edge, interior.' }
    }
    tentative.load_location = v
  } else {
    const n = toFiniteNumber(normalizedValue)
    if (n === null) {
      return { ok: false, normalizedValue, error: 'Enter a numeric value.' }
    }
    tentative[key] = n
  }

  const a = toFiniteNumber(tentative.a)
  const b = toFiniteNumber(tentative.b)
  const t = toFiniteNumber(tentative.t)

  const x1 = toFiniteNumber(tentative.x1)
  const x2 = toFiniteNumber(tentative.x2)
  const y1 = toFiniteNumber(tentative.y1)
  const y2 = toFiniteNumber(tentative.y2)

  // Core slab validations
  if (key === 'a' || key === 'b' || key === 't') {
    if (key === 'a' && !isFinitePositive(a)) {
      return { ok: false, normalizedValue, error: 'Slab length must be positive.' }
    }
    if (key === 'b' && !isFinitePositive(b)) {
      return { ok: false, normalizedValue, error: 'Slab width must be positive.' }
    }
    if (key === 't') {
      if (!isFinitePositive(t)) {
        return { ok: false, normalizedValue, error: 'Slab thickness must be positive.' }
      }
      if (isFinitePositive(a) && t >= a) {
        return { ok: false, normalizedValue, error: 'Slab thickness must be less than slab length.' }
      }
      if (isFinitePositive(b) && t >= b) {
        return { ok: false, normalizedValue, error: 'Slab thickness must be less than slab width.' }
      }
    }
  }

  // Additional cross-checks when a/b change.
  if (key === 'a' && isFinitePositive(a)) {
    if (isFinitePositive(t) && t >= a) {
      return { ok: false, normalizedValue, error: 'Slab thickness must be less than slab length.' }
    }
    if (isFiniteNonNegative(x1) && x1 > a) {
      return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab length.' }
    }
    if (isFiniteNonNegative(x2) && x2 > a) {
      return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab length.' }
    }
  }

  if (key === 'b' && isFinitePositive(b)) {
    if (isFinitePositive(t) && t >= b) {
      return { ok: false, normalizedValue, error: 'Slab thickness must be less than slab width.' }
    }
    if (isFiniteNonNegative(y1) && y1 > b) {
      return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab width.' }
    }
    if (isFiniteNonNegative(y2) && y2 > b) {
      return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab width.' }
    }
  }

  // nu validation with warning
  if (key === 'nu') {
    const nu = toFiniteNumber(tentative.nu)
    if (nu === null) {
      return { ok: false, normalizedValue, error: 'Enter a numeric value.' }
    }
    if (nu < 0.15 || nu > 0.25) {
      return { ok: false, normalizedValue, error: "Poisson's ratio must be between 0.15 and 0.25 (IS 456)." }
    }
    return { ok: true, normalizedValue: String(nu) }
  }

  if (key === 'Emod') {
    const emod = toFiniteNumber(tentative.Emod)
    if (emod === null || emod <= 1000) {
      return { ok: false, normalizedValue, error: 'Elastic modulus must be greater than 1000 MPa.' }
    }
  }

  if (key === 'Kx' || key === 'Ky' || key === 'Kz') {
    const kval = toFiniteNumber(tentative[key])
    if (kval === null || kval < 0) {
      return { ok: false, normalizedValue, error: 'Soil stiffness must be ≥ 0.' }
    }
  }

  // Load ordering and bounds
  if (key === 'x1' || key === 'x2') {
    const nx1 = toFiniteNumber(tentative.x1)
    const nx2 = toFiniteNumber(tentative.x2)
    if (nx1 !== null && nx2 !== null && nx1 >= nx2) {
      return { ok: false, normalizedValue, error: 'Load start must be less than load end.' }
    }
    if (isFinitePositive(a)) {
      if (nx1 !== null && (nx1 < 0 || nx1 > a)) {
        return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab dimensions.' }
      }
      if (nx2 !== null && (nx2 < 0 || nx2 > a)) {
        return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab dimensions.' }
      }
    }
  }

  if (key === 'y1' || key === 'y2') {
    const ny1 = toFiniteNumber(tentative.y1)
    const ny2 = toFiniteNumber(tentative.y2)
    if (ny1 !== null && ny2 !== null && ny1 >= ny2) {
      return { ok: false, normalizedValue, error: 'Load start must be less than load end.' }
    }
    if (isFinitePositive(b)) {
      if (ny1 !== null && (ny1 < 0 || ny1 > b)) {
        return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab dimensions.' }
      }
      if (ny2 !== null && (ny2 < 0 || ny2 > b)) {
        return { ok: false, normalizedValue, error: 'Load coordinates must be inside slab dimensions.' }
      }
    }
  }

  if (key === 'q') {
    const q = toFiniteNumber(tentative.q)
    if (q === null || q <= 0) {
      return { ok: false, normalizedValue, error: 'Tire pressure must be positive.' }
    }
  }

  // Generic positive checks for core geometry/material keys.
  if ((key === 'a' || key === 'b' || key === 't') && toFiniteNumber(tentative[key]) !== null) {
    const n = toFiniteNumber(tentative[key])!
    if (n <= 0) {
      return { ok: false, normalizedValue, error: 'This value must be greater than zero.' }
    }
  }

  // If we reached here, it is valid.
  // Normalize enums to lower-case.
  if (key === 'mesh_type' || key === 'load_location') {
    return { ok: true, normalizedValue: clampLowered(normalizedValue) }
  }

  const finalNum = toFiniteNumber(tentative[key])
  return { ok: true, normalizedValue: finalNum === null ? normalizedValue : String(finalNum) }
}
