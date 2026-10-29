import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# =========================
# IO
# =========================
def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def write_json(path: str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


# =========================
# Basic helpers
# =========================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))



def as_bool(x: Any) -> bool:
    return bool(x)



def get_candidate_id(c: Dict[str, Any]) -> str:
    cid = c.get('candidate_id')
    if not cid:
        raise ValueError(f'Missing candidate_id in row: {c}')
    return str(cid)



def derive_type(c: Dict[str, Any]) -> str:
    return str(c.get('type', 'unknown'))



def get_total_constraints(c: Dict[str, Any]) -> int:
    total = c.get('total')
    if total is not None:
        try:
            return int(total)
        except Exception:
            pass
    return len(c.get('constraints', []) or [])



def get_cell_key(c: Dict[str, Any]) -> Tuple[str, int]:
    return derive_type(c), get_total_constraints(c)



def has_negation(c: Dict[str, Any]) -> bool:
    return as_bool(c.get('has_negation', False))



def get_support(c: Dict[str, Any]) -> int:
    return int(c.get('exact_support_count', c.get('support', 0)) or 0)



def get_fixed_constraint_missing_1(c: Dict[str, Any]) -> int:
    return int(c.get('fixed_constraint_missing_1_count', 0) or 0)



def get_fixed_constraint_mismatch_1(c: Dict[str, Any]) -> int:
    return int(c.get('fixed_constraint_mismatch_1_count', 0) or 0)



def get_fixed_constraint_near_miss_1(c: Dict[str, Any]) -> int:
    total = c.get('fixed_constraint_near_miss_1_count')
    if total is not None:
        return int(total or 0)
    return get_fixed_constraint_missing_1(c) + get_fixed_constraint_mismatch_1(c)



def get_negated_field_missing(c: Dict[str, Any]) -> int:
    return int(c.get('negated_field_missing_count', c.get('negated_field_missing', 0)) or 0)



def get_negated_field_mismatch(c: Dict[str, Any]) -> int:
    return int(c.get('negated_field_mismatch_count', c.get('negated_field_mismatch', 0)) or 0)



def get_explore_field_missing_count(c: Dict[str, Any]) -> int:
    return int(c.get('explore_field_missing_count', c.get('explore_field_missing', 0)) or 0)



def get_explore_field_missing_ratio(c: Dict[str, Any]) -> float:
    x = c.get('explore_field_missing_ratio', 0.0)
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0



def get_distinct_explored_value_count(c: Dict[str, Any]) -> int:
    return int(c.get('distinct_explored_value_count', 0) or 0)



def get_informative_boundary(c: Dict[str, Any]) -> int:
    return (
        get_fixed_constraint_near_miss_1(c)
        + get_negated_field_missing(c)
        + get_negated_field_mismatch(c)
        + get_explore_field_missing_count(c)
    )



def get_informative_boundary_per_support(c: Dict[str, Any]) -> float:
    support = max(get_support(c), 1)
    return float(get_informative_boundary(c)) / float(support)



def get_constraint_signature(c: Dict[str, Any]) -> Tuple[Tuple[str, str, str], ...]:
    items = []
    for it in c.get('constraints', []) or []:
        items.append((str(it.get('field', '')), str(it.get('op', '')), str(it.get('value', ''))))
    return tuple(sorted(items))



def get_support_id_set(c: Dict[str, Any]) -> Set[str]:
    return {str(x) for x in (c.get('support_paragraph_ids', []) or [])}



def is_exploration_type(type_name: str) -> bool:
    return 'exploration' in type_name



def merge_unique_rows(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows_a + rows_b:
        if isinstance(row, dict) and row.get('candidate_id'):
            out[str(row['candidate_id'])] = row
    return list(out.values())


# =========================
# State helpers
# =========================
def _load_id_like_file(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()

    if p.suffix == '.jsonl':
        rows = read_jsonl(str(p))
        out: Set[str] = set()
        for r in rows:
            if isinstance(r, dict) and r.get('candidate_id'):
                out.add(str(r['candidate_id']))
            elif isinstance(r, str):
                out.add(r)
        return out

    if p.suffix == '.json':
        obj = read_json(str(p))
        out: Set[str] = set()
        if isinstance(obj, list):
            for r in obj:
                if isinstance(r, dict) and r.get('candidate_id'):
                    out.add(str(r['candidate_id']))
                elif isinstance(r, str):
                    out.add(r)
        elif isinstance(obj, dict) and isinstance(obj.get('candidate_ids'), list):
            out.update(str(x) for x in obj['candidate_ids'])
        return out

    out: Set[str] = set()
    with open(str(p), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line)
    return out


# =========================
# Target helpers
# =========================
def normalize_ratio_map(ratio_map: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in ratio_map.values())
    if total <= 0:
        n = max(1, len(ratio_map))
        return {k: 1.0 / n for k in ratio_map}
    return {k: float(v) / total for k, v in ratio_map.items()}



def largest_remainder_counts(total: int, ratio_map: Dict[str, float]) -> Dict[str, int]:
    keys = list(ratio_map.keys())
    if total <= 0 or not keys:
        return {k: 0 for k in keys}
    raw = {k: total * float(ratio_map[k]) for k in keys}
    base = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(base.values())
    order = sorted(keys, key=lambda k: (raw[k] - base[k], raw[k]), reverse=True)
    for k in order[:remainder]:
        base[k] += 1
    return base



def compute_type_targets(total_target: int, cfg: Dict[str, Any]) -> Dict[str, int]:
    return largest_remainder_counts(total_target, normalize_ratio_map(cfg['type_ratio']))



def get_existing_totals_by_type(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, List[int]]:
    allowed_totals = {int(x) for x in cfg.get('totals', [3, 4, 5])}
    out: Dict[str, Set[int]] = defaultdict(set)
    for c in rows:
        t = derive_type(c)
        total = get_total_constraints(c)
        if total in allowed_totals:
            out[t].add(total)
    return {k: sorted(v) for k, v in out.items()}



def get_total_ratio_for_type(type_name: str, cfg: Dict[str, Any], existing_totals: Optional[List[int]] = None) -> Dict[str, float]:
    ratio_cfg = cfg.get('total_ratio_by_type', {})
    if existing_totals:
        existing_keys = {str(x) for x in existing_totals}
    else:
        existing_keys = None

    if isinstance(ratio_cfg.get(type_name), dict):
        raw = {str(k): float(v) for k, v in ratio_cfg[type_name].items()}
        if existing_keys is not None:
            raw = {k: v for k, v in raw.items() if k in existing_keys}
        if raw:
            return normalize_ratio_map(raw)

    totals = existing_totals if existing_totals else cfg.get('totals', [3, 4, 5])
    return normalize_ratio_map({str(x): 1.0 for x in totals})



def compute_cell_targets(total_target: int, cfg: Dict[str, Any], available_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    type_targets = compute_type_targets(total_target, cfg)
    existing_totals_by_type = get_existing_totals_by_type(available_rows, cfg)
    out: Dict[str, int] = {}
    for type_name, type_target in type_targets.items():
        existing_totals = existing_totals_by_type.get(type_name, [])
        if not existing_totals:
            continue
        total_counts = largest_remainder_counts(type_target, get_total_ratio_for_type(type_name, cfg, existing_totals))
        for total_str, cnt in total_counts.items():
            out[f'{type_name}__{int(total_str)}'] = cnt
    return out



def count_by_cell(cands: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cands:
        cell_key = f'{derive_type(c)}__{get_total_constraints(c)}'
        out[cell_key] = out.get(cell_key, 0) + 1
    return out



def cell_deficits(accepted: List[Dict[str, Any]], cell_targets: Dict[str, int]) -> Dict[str, int]:
    current = count_by_cell(accepted)
    return {k: max(int(v) - int(current.get(k, 0)), 0) for k, v in cell_targets.items()}



def choose_cell_to_fill(accepted_like: List[Dict[str, Any]], available_by_cell: Dict[str, List[Dict[str, Any]]], cell_targets: Dict[str, int]) -> Optional[str]:
    deficits = cell_deficits(accepted_like, cell_targets)
    choices = []
    for cell_key, deficit in deficits.items():
        if deficit <= 0:
            continue
        if not available_by_cell.get(cell_key):
            continue
        target = max(1, int(cell_targets.get(cell_key, 1)))
        choices.append((deficit / target, deficit, cell_key))
    if not choices:
        return None
    choices.sort(reverse=True)
    return choices[0][2]


# =========================
# Hard filter + relaxed fallback
# =========================
def get_hard_filter_cfg_for_candidate(c: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = cfg.get('hard_filters', {})
    out = dict(root.get('default', {}))
    by_type = root.get('by_type', {})
    out.update(by_type.get(derive_type(c), {}))
    return out



def compute_violation_details(c: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    hf = get_hard_filter_cfg_for_candidate(c, cfg)
    support = max(get_support(c), 1)
    fixed_constraint_near_miss_total = get_fixed_constraint_near_miss_1(c)
    fixed_constraint_mismatch_per_support = get_fixed_constraint_mismatch_1(c) / support
    informative_boundary = get_informative_boundary(c)
    informative_boundary_per_support = informative_boundary / support

    failed_checks = 0
    score = 0.0
    reasons: List[Dict[str, Any]] = []

    def add_failure(name: str, actual: float, expected: float, kind: str, weight: float, base: float):
        nonlocal failed_checks, score
        failed_checks += 1
        delta = max(float(base), abs(float(actual) - float(expected)))
        score += float(weight) * delta / float(base)
        reasons.append({
            'name': name,
            'kind': kind,
            'actual': actual,
            'expected': expected,
            'weight': weight,
        })

    if hf.get('min_support') is not None and support < int(hf['min_support']):
        add_failure('min_support', support, int(hf['min_support']), 'min', 1.25, max(1.0, float(hf['min_support'])))

    if hf.get('max_support') is not None and support > int(hf['max_support']):
        add_failure('max_support', support, int(hf['max_support']), 'max', 0.75, max(1.0, float(hf['max_support'])))

    if hf.get('min_fixed_constraint_near_miss') is not None and fixed_constraint_near_miss_total < int(hf['min_fixed_constraint_near_miss']):
        add_failure(
            'min_fixed_constraint_near_miss',
            fixed_constraint_near_miss_total,
            int(hf['min_fixed_constraint_near_miss']),
            'min',
            0.9,
            max(1.0, float(hf['min_fixed_constraint_near_miss'])),
        )

    if hf.get('max_fixed_constraint_near_miss') is not None and fixed_constraint_near_miss_total > int(hf['max_fixed_constraint_near_miss']):
        add_failure(
            'max_fixed_constraint_near_miss',
            fixed_constraint_near_miss_total,
            int(hf['max_fixed_constraint_near_miss']),
            'max',
            1.2,
            max(1.0, float(hf['max_fixed_constraint_near_miss'])),
        )

    if hf.get('max_fixed_constraint_mismatch_per_support') is not None and fixed_constraint_mismatch_per_support > float(hf['max_fixed_constraint_mismatch_per_support']):
        add_failure(
            'max_fixed_constraint_mismatch_per_support',
            fixed_constraint_mismatch_per_support,
            float(hf['max_fixed_constraint_mismatch_per_support']),
            'max',
            1.0,
            max(1.0, float(hf['max_fixed_constraint_mismatch_per_support'])),
        )

    if hf.get('min_informative_boundary') is not None and informative_boundary < int(hf['min_informative_boundary']):
        add_failure('min_informative_boundary', informative_boundary, int(hf['min_informative_boundary']), 'min', 1.0, max(1.0, float(hf['min_informative_boundary'])))

    if hf.get('min_informative_boundary_per_support') is not None and informative_boundary_per_support < float(hf['min_informative_boundary_per_support']):
        add_failure(
            'min_informative_boundary_per_support',
            informative_boundary_per_support,
            float(hf['min_informative_boundary_per_support']),
            'min',
            1.1,
            max(0.5, float(hf['min_informative_boundary_per_support'])),
        )

    min_neg_spec = hf.get('min_negation_specific_boundary')
    if has_negation(c) and min_neg_spec is not None:
        neg_spec = get_negated_field_missing(c) + get_negated_field_mismatch(c)
        if neg_spec < int(min_neg_spec):
            add_failure('min_negation_specific_boundary', neg_spec, int(min_neg_spec), 'min', 0.8, max(1.0, float(min_neg_spec)))

    return {
        'pass': failed_checks == 0,
        'failed_checks': failed_checks,
        'violation_score': round(score, 6),
        'reasons': reasons,
    }



def assign_filter_tier(c: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    details = compute_violation_details(c, cfg)
    relaxed_cfg = cfg.get('relaxed_fallback', {})
    enabled = bool(relaxed_cfg.get('enabled', True))
    max_score = float(relaxed_cfg.get('max_relaxed_violation_score', 2.5))
    max_failed = int(relaxed_cfg.get('max_relaxed_failed_checks', 2))

    if details['pass']:
        tier = 'hard_pass'
    elif enabled and details['failed_checks'] <= max_failed and details['violation_score'] <= max_score:
        tier = 'relaxed_pass'
    else:
        tier = 'bad_fail'

    out = dict(details)
    out['tier'] = tier
    return out


# =========================
# Scoring
# =========================
def plateau_score(x: float, ideal_min: float, ideal_max: float, zero_below: Optional[float], zero_above: Optional[float]) -> float:
    x = float(x)
    ideal_min = float(ideal_min)
    ideal_max = float(ideal_max)
    if ideal_min > ideal_max:
        ideal_min, ideal_max = ideal_max, ideal_min
    if zero_below is None:
        zero_below = ideal_min
    if zero_above is None:
        zero_above = ideal_max
    zero_below = float(zero_below)
    zero_above = float(zero_above)

    if x < zero_below or x > zero_above:
        return 0.0
    if ideal_min <= x <= ideal_max:
        return 1.0
    if x < ideal_min:
        denom = max(ideal_min - zero_below, 1e-9)
        return clamp01((x - zero_below) / denom)
    denom = max(zero_above - ideal_max, 1e-9)
    return clamp01((zero_above - x) / denom)



def scaled_linear_count(x: int, cap: int) -> float:
    cap = max(1, int(cap))
    return clamp01(min(max(0, int(x)), cap) / cap)



def quality_score(c: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    norm = cfg.get('normalization', {})
    type_name = derive_type(c)

    support_score = plateau_score(
        x=get_support(c),
        ideal_min=norm.get('support_ideal_min', 3),
        ideal_max=norm.get('support_ideal_max', 6),
        zero_below=norm.get('support_floor', 1),
        zero_above=norm.get('support_zero_after', 12),
    )
    fixed_constraint_near_miss_score = plateau_score(
        x=get_fixed_constraint_near_miss_1(c),
        ideal_min=norm.get('fixed_constraint_near_miss_ideal_min', 5),
        ideal_max=norm.get('fixed_constraint_near_miss_ideal_max', 50),
        zero_below=norm.get('fixed_constraint_near_miss_floor', 0),
        zero_above=norm.get('fixed_constraint_near_miss_zero_after', 200),
    )

    if is_exploration_type(type_name):
        weights = cfg.get('quality_weights', {}).get('exploration', {})
        distinct_score = scaled_linear_count(get_distinct_explored_value_count(c), norm.get('distinct_cap', 8))
        explore_field_presence_quality = 1.0 - clamp01(get_explore_field_missing_ratio(c))
        parts = {
            'fixed_constraint_near_miss_quality': float(weights.get('fixed_constraint_near_miss_quality', 0.0)) * fixed_constraint_near_miss_score,
            'support_quality': float(weights.get('support_quality', 0.0)) * support_score,
            'distinct_explored_value_count': float(weights.get('distinct_explored_value_count', 0.0)) * distinct_score,
            'explore_field_missing_quality': float(weights.get('explore_field_missing_quality', 0.0)) * explore_field_presence_quality,
        }
        return sum(parts.values()), parts

    weights = cfg.get('quality_weights', {}).get('reproduction', {})
    parts = {
        'fixed_constraint_near_miss_quality': float(weights.get('fixed_constraint_near_miss_quality', 0.0)) * fixed_constraint_near_miss_score,
        'support_quality': float(weights.get('support_quality', 0.0)) * support_score,
    }
    return sum(parts.values()), parts


# =========================
# Redundancy
# =========================
def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)



def similarity_score(a: Dict[str, Any], b: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    weights = cfg.get('redundancy', {}).get('weights', {})
    cons_sim = jaccard(set(get_constraint_signature(a)), set(get_constraint_signature(b)))
    support_sim = jaccard(get_support_id_set(a), get_support_id_set(b))
    score = (
        float(weights.get('constraint_overlap', 0.6)) * cons_sim
        + float(weights.get('support_overlap', 0.4)) * support_sim
    )
    return min(score, 1.0)



def scope_filtered_selected(candidate: Dict[str, Any], selected: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    if scope == 'all':
        return selected
    if scope == 'same_type':
        t = derive_type(candidate)
        return [x for x in selected if derive_type(x) == t]
    if scope == 'same_cell':
        cell = get_cell_key(candidate)
        return [x for x in selected if get_cell_key(x) == cell]
    raise ValueError(f'Unsupported redundancy scope: {scope}')



def exact_duplicate(candidate: Dict[str, Any], selected: List[Dict[str, Any]], scope: str) -> bool:
    scoped = scope_filtered_selected(candidate, selected, scope)
    sig0 = (
        derive_type(candidate),
        get_total_constraints(candidate),
        candidate.get('explore_field'),
        candidate.get('negated_field'),
        get_constraint_signature(candidate),
    )
    for s in scoped:
        sig1 = (
            derive_type(s),
            get_total_constraints(s),
            s.get('explore_field'),
            s.get('negated_field'),
            get_constraint_signature(s),
        )
        if sig0 == sig1:
            return True
    return False



def redundancy_penalty(candidate: Dict[str, Any], selected: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[float, float]:
    red = cfg.get('redundancy', {})
    scope = red.get('scope', 'same_cell')
    scoped = scope_filtered_selected(candidate, selected, scope)
    if not scoped:
        return 0.0, 0.0
    max_sim = max(similarity_score(candidate, s, cfg) for s in scoped)
    penalty_strength = float(red.get('penalty_strength', 0.35))
    return penalty_strength * max_sim, max_sim


# =========================
# Candidate preparation
# =========================
def prepare_candidates(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        c = dict(row)
        filter_eval = assign_filter_tier(c, cfg)
        c['_derived_type'] = derive_type(c)
        c['_derived_total'] = get_total_constraints(c)
        c['_derived_cell_key'] = f"{c['_derived_type']}__{c['_derived_total']}"
        c['_filter_eval'] = filter_eval
        out.append(c)
    return out


# =========================
# Selection helpers
# =========================
def select_one_candidate(candidates: List[Dict[str, Any]], selected_reference: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    scope = cfg.get('redundancy', {}).get('scope', 'same_cell')
    scored_rows = []

    for c in candidates:
        if exact_duplicate(c, selected_reference, scope):
            continue

        q_score, q_parts = quality_score(c, cfg)
        red_penalty, max_sim = redundancy_penalty(c, selected_reference, cfg)

        filter_eval = c.get('_filter_eval', {})
        tier = filter_eval.get('tier', 'hard_pass')
        violation_score = float(filter_eval.get('violation_score', 0.0))
        failed_checks = int(filter_eval.get('failed_checks', 0))

        base_tier_penalty = 0.0 if tier == 'hard_pass' else (0.10 if tier == 'relaxed_pass' else 0.30)
        violation_penalty = base_tier_penalty + 0.04 * violation_score + 0.03 * failed_checks
        final_score = q_score - red_penalty - violation_penalty

        scored_rows.append((
            final_score,
            q_score,
            -red_penalty,
            -violation_penalty,
            get_candidate_id(c),
            c,
            q_parts,
            max_sim,
        ))

    if not scored_rows:
        return None

    scored_rows.sort(reverse=True)
    _, q_score, neg_red_penalty, neg_violation_penalty, _, chosen, q_parts, max_sim = scored_rows[0]
    out = dict(chosen)
    out['_sampling'] = {
        'quality_score': q_score,
        'redundancy_penalty': -neg_red_penalty,
        'violation_penalty': -neg_violation_penalty,
        'max_similarity_to_selected': max_sim,
        'final_score': q_score + neg_red_penalty + neg_violation_penalty,
        'quality_parts': q_parts,
        'filter_eval': out.get('_filter_eval', {}),
    }
    return out



def summarize_pool_by_cell(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hard_pass': 0, 'relaxed_pass': 0, 'bad_fail': 0, 'all': 0})
    for c in rows:
        key = c.get('_derived_cell_key', f"{derive_type(c)}__{get_total_constraints(c)}")
        tier = c.get('_filter_eval', {}).get('tier', 'hard_pass')
        out[key]['all'] += 1
        out[key][tier] += 1
    return dict(out)



def summarize_type_total(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in rows:
        out[derive_type(c)][str(get_total_constraints(c))] += 1
    return {k: dict(v) for k, v in out.items()}


# =========================
# Core selection
# =========================
def incremental_sample(pool_rows: List[Dict[str, Any]], sampled_ids: Set[str], rejected_ids: Set[str], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool = prepare_candidates(pool_rows, cfg)
    pool_by_id = {get_candidate_id(c): c for c in pool}

    accepted_ids = {cid for cid in sampled_ids if cid not in rejected_ids and cid in pool_by_id}
    used_ids = set(sampled_ids) | set(rejected_ids)

    accepted = [pool_by_id[cid] for cid in sorted(accepted_ids) if cid in pool_by_id]
    available = [c for c in pool if get_candidate_id(c) not in used_ids]

    total_target = int(cfg['total_target'])
    batch_size = int(cfg.get('selection', {}).get('batch_size', total_target))
    allow_bad_fallback = bool(cfg.get('relaxed_fallback', {}).get('allow_bad_fallback', False))
    cell_targets = compute_cell_targets(total_target, cfg, available)

    manifest: Dict[str, Any] = {
        'total_pool_size': len(pool),
        'sampled_ids_count': len(sampled_ids),
        'rejected_ids_count': len(rejected_ids),
        'accepted_count_before': len(accepted),
        'available_count_before_filtering': len(available),
        'cell_targets': cell_targets,
        'accepted_cell_counts_before': count_by_cell(accepted),
        'available_pool_by_cell_and_tier': summarize_pool_by_cell(available),
        'available_pool_type_total': summarize_type_total(available),
        'existing_totals_by_type': get_existing_totals_by_type(available, cfg),
        'blocked_cells': [],
        'steps': [],
    }

    selected_this_round: List[Dict[str, Any]] = []
    selected_ids_this_round: Set[str] = set()
    blocked_cells: Set[str] = set()

    for step_idx in range(batch_size):
        accepted_like = accepted + selected_this_round
        available_now = [c for c in available if get_candidate_id(c) not in selected_ids_this_round]
        if not available_now:
            break

        available_by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in available_now:
            if c['_derived_cell_key'] in blocked_cells:
                continue
            available_by_cell[c['_derived_cell_key']].append(c)

        cell_key = choose_cell_to_fill(accepted_like, available_by_cell, cell_targets)
        if cell_key is None:
            break

        in_cell = list(available_by_cell.get(cell_key, []))
        hard_pass = [c for c in in_cell if c['_filter_eval']['tier'] == 'hard_pass']
        relaxed_pass = [c for c in in_cell if c['_filter_eval']['tier'] == 'relaxed_pass']
        bad_fail = [c for c in in_cell if c['_filter_eval']['tier'] == 'bad_fail']

        chosen = select_one_candidate(hard_pass, accepted_like, cfg)
        used_tier = 'hard_pass'

        if chosen is None and relaxed_pass:
            chosen = select_one_candidate(relaxed_pass, accepted_like, cfg)
            used_tier = 'relaxed_pass'

        if chosen is None and allow_bad_fallback and bad_fail:
            chosen = select_one_candidate(bad_fail, accepted_like, cfg)
            used_tier = 'bad_fail'

        if chosen is None:
            blocked_cells.add(cell_key)
            manifest['steps'].append({
                'step': step_idx + 1,
                'selected_candidate_id': None,
                'cell_key': cell_key,
                'selection_tier': None,
                'status': 'blocked_no_selectable_candidate',
                'cell_pool_counts': {
                    'hard_pass': len(hard_pass),
                    'relaxed_pass': len(relaxed_pass),
                    'bad_fail': len(bad_fail),
                },
            })
            continue

        cid = get_candidate_id(chosen)
        selected_this_round.append(chosen)
        selected_ids_this_round.add(cid)

        manifest['steps'].append({
            'step': step_idx + 1,
            'selected_candidate_id': cid,
            'cell_key': cell_key,
            'type': derive_type(chosen),
            'total': get_total_constraints(chosen),
            'selection_tier': used_tier,
            'status': 'selected',
            'sampling': chosen.get('_sampling', {}),
        })

    accepted_after = accepted + selected_this_round
    manifest['accepted_count_after'] = len(accepted_after)
    manifest['selected_this_round_count'] = len(selected_this_round)
    manifest['accepted_cell_counts_after'] = count_by_cell(accepted_after)
    manifest['cell_deficits_after'] = cell_deficits(accepted_after, cell_targets)
    manifest['selected_type_total'] = summarize_type_total(selected_this_round)
    manifest['blocked_cells'] = sorted(blocked_cells)

    return selected_this_round, manifest


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Incremental sampler using fixed/negated/explore boundary statistics.')
    parser.add_argument('--config', default='seed_data/sample_config.json')
    parser.add_argument('--domains', nargs='*', default=['polymer_materials'])
    parser.add_argument('--pool-template', default='seed_data/{domain}/constraints/constraint_candidates_with_boundary_stats.jsonl')
    parser.add_argument('--rejected-template', default='seed_data/{domain}/constraints/rejected_constraints_stage2.jsonl')
    parser.add_argument('--sampled-template', default='seed_data/{domain}/constraints/sampled_constraints_stage1.jsonl')
    parser.add_argument('--out-jsonl-template', default='seed_data/{domain}/constraints/sampled_constraints_stage1.jsonl')
    parser.add_argument('--out-manifest-template', default='seed_data/{domain}/constraints/out_manifest.json')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    write_mode = str(cfg.get('selection', {}).get('write_mode', 'supplement')).lower()

    for domain in args.domains:
        pool_path = args.pool_template.format(domain=domain)
        rejected_path = args.rejected_template.format(domain=domain)
        sampled_path = args.sampled_template.format(domain=domain)
        out_jsonl_path = args.out_jsonl_template.format(domain=domain)
        out_manifest_path = args.out_manifest_template.format(domain=domain)

        pool_rows = read_jsonl(pool_path)
        sampled_ids = set() if write_mode == 'overwrite' else _load_id_like_file(sampled_path)
        rejected_ids = _load_id_like_file(rejected_path)

        selected, manifest = incremental_sample(
            pool_rows=pool_rows,
            sampled_ids=sampled_ids,
            rejected_ids=rejected_ids,
            cfg=cfg,
        )

        if write_mode == 'overwrite':
            rows_to_write = selected
        else:
            existing_rows = read_jsonl(sampled_path)
            rows_to_write = merge_unique_rows(existing_rows, selected)

        write_jsonl(out_jsonl_path, rows_to_write)
        write_json(out_manifest_path, manifest)

        print(f'[OK] Domain={domain} | New selected={len(selected)} | Wrote total rows={len(rows_to_write)} -> {out_jsonl_path}')
        print(f'[OK] Wrote manifest -> {out_manifest_path}')


if __name__ == '__main__':
    main()



# conda activate retrieve_dataset
# python scripts/constraint_set/sample_final_constraints.py
# python scripts/constraint_set/sample_final_constraints.py --domains cell_culture
# python scripts/constraint_set/sample_final_constraints.py --domains polymer_materials
# python scripts/constraint_set/sample_final_constraints.py --domains electrochemistry
# python scripts/constraint_set/sample_final_constraints.py --domains immunotechnology
# python scripts/constraint_set/sample_final_constraints.py --domains inorganic_materials
# python scripts/constraint_set/sample_final_constraints.py --domains organic_synthesis
# python scripts/constraint_set/sample_final_constraints.py --domains electrochemistry cell_culture immunotechnology inorganic_materials organic_synthesis polymer_materials
