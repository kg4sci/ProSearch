import json
import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
from typing import Any, Dict, List

from tqdm import tqdm

EPS = 1e-8
G_SETUPS = None
G_FIELDS_CFG = None
G_FIELD_NONEMPTY_INDEX = None
G_EXACT_VALUE_INDEX = None
G_LITERAL_MATCH_CACHE = None


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_template_combinations(path):
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ['combinations', 'templates', 'items', 'data']:
            if k in data and isinstance(data[k], list):
                return data[k]
    raise ValueError(f'Unsupported template combinations format: {path}')


def normalize_unit(unit):
    if unit is None:
        return None
    unit = str(unit).strip()
    if not unit:
        return None
    unit = unit.replace('µ', 'μ').replace('ug', 'μg').replace('mcg', 'μg')
    return unit


def is_numeric_field(field_cfg):
    return field_cfg.get('value_type') == 'numeric'


def literal_sort_key(lit):
    return (lit['field'], lit['op'], json.dumps(lit, ensure_ascii=False, sort_keys=True))


def literal_key(lit):
    if lit['op'] in ('=', '!='):
        payload = {'field': lit['field'], 'op': lit['op'], 'value': lit['value']}
        if 'parsed' in lit:
            payload['parsed'] = lit['parsed']
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if lit['op'] == 'between':
        return json.dumps({'field': lit['field'], 'op': 'between', 'range': lit['range']}, ensure_ascii=False, sort_keys=True)
    raise ValueError(f'Unsupported op: {lit["op"]}')


def candidate_key(candidate):
    payload = {
        'domain': candidate['domain'],
        'type': candidate['type'],
        'type_id': candidate['type_id'],
        'total': candidate['total'],
        'explore_field': candidate.get('explore_field'),
        'constraints': sorted(candidate['constraints'], key=literal_sort_key),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def dedupe_literals(literals):
    seen = set()
    out = []
    for lit in literals:
        k = literal_key(lit)
        if k not in seen:
            seen.add(k)
            out.append(lit)
    return out


def get_explore_field(template):
    return template.get('field_assignment', {}).get('Explore')


def units_compatible(u1, u2):
    u1 = normalize_unit(u1)
    u2 = normalize_unit(u2)
    if u1 is None or u2 is None:
        return True
    return u1 == u2


def eq_number(a, b, eps=EPS):
    return abs(float(a) - float(b)) <= eps


def numeric_literal_matches_entry(literal, entry):
    sample_parsed = entry.get('parsed')
    if not sample_parsed:
        return False

    if literal['op'] == '=' and 'parsed' in literal:
        q_parsed = literal['parsed']
        q_kind = q_parsed.get('kind')
        s_kind = sample_parsed.get('kind')
        if q_kind == 'ratio':
            return entry.get('value') == literal.get('value')
        if q_kind == 'point':
            q = q_parsed['number']
            if not units_compatible(q_parsed.get('unit'), sample_parsed.get('unit')):
                return False
            if s_kind == 'point':
                return eq_number(sample_parsed['number'], q)
            if s_kind == 'range':
                return sample_parsed['lower'] - EPS <= q <= sample_parsed['upper'] + EPS
        return False

    if literal['op'] == 'between':
        ql = literal['range']['lower']
        qr = literal['range']['upper']
        qu = literal['range'].get('unit')
        if not units_compatible(qu, sample_parsed.get('unit')):
            return False
        s_kind = sample_parsed.get('kind')
        if s_kind == 'point':
            d = sample_parsed['number']
            return ql - EPS <= d <= qr + EPS
        if s_kind == 'range':
            dl = sample_parsed['lower']
            dr = sample_parsed['upper']
            return ql - EPS <= dl and dr <= qr + EPS
        return False

    return False


def literal_matches_entry(literal, entry):
    if literal['op'] == '=' and 'parsed' not in literal and 'range' not in literal:
        return entry.get('value') == literal['value']
    if literal['op'] == '=' and 'parsed' in literal:
        return numeric_literal_matches_entry(literal, entry)
    if literal['op'] == 'between':
        return numeric_literal_matches_entry(literal, entry)
    return False


def any_entry_matches_literal(entries, literal):
    return any(literal_matches_entry(literal, e) for e in entries)


def build_indexes(setup_records):
    field_nonempty_index = defaultdict(set)
    exact_value_index = defaultdict(lambda: defaultdict(set))
    for i, rec in enumerate(setup_records):
        flat_values = rec.get('flat_values', {})
        for field, entries in flat_values.items():
            if entries:
                field_nonempty_index[field].add(i)
            for e in entries:
                v = e.get('value') if isinstance(e, dict) else None
                if v is not None:
                    exact_value_index[field][v].add(i)
    return field_nonempty_index, exact_value_index


def init_worker(setup_records_path, field_registry_path):
    global G_SETUPS, G_FIELDS_CFG, G_FIELD_NONEMPTY_INDEX, G_EXACT_VALUE_INDEX, G_LITERAL_MATCH_CACHE
    G_SETUPS = list(read_jsonl(setup_records_path))
    G_FIELDS_CFG = read_json(field_registry_path)
    G_FIELD_NONEMPTY_INDEX, G_EXACT_VALUE_INDEX = build_indexes(G_SETUPS)
    G_LITERAL_MATCH_CACHE = {}


def build_literal_from_entry(field, entry, field_cfg):
    value = entry['value']
    if not is_numeric_field(field_cfg):
        return {'field': field, 'op': '=', 'value': value}
    parsed = entry.get('parsed')
    if not parsed:
        return {'field': field, 'op': '=', 'value': value}
    if parsed.get('kind') == 'range':
        return {
            'field': field,
            'op': 'between',
            'range': {
                'lower': parsed['lower'],
                'upper': parsed['upper'],
                'unit': normalize_unit(parsed.get('unit')),
            },
            'surface': value,
        }
    return {'field': field, 'op': '=', 'value': value, 'parsed': parsed}


def infer_triggered_binding_groups(setup_record, constraints):
    item_bindings = setup_record.get('item_bindings', {}) or {}
    fields_in_constraints = {lit['field'] for lit in constraints}
    triggered = []
    for list_path, items in item_bindings.items():
        group_fields = set()
        for item in items:
            group_fields.update(item.keys())
        overlap = fields_in_constraints.intersection(group_fields)
        if len(overlap) >= 2:
            triggered.append({
                'list_path': list_path,
                'fields': sorted(group_fields),
            })
    return triggered


def instantiate_candidates_from_setup(setup_record, template):
    flat_values = setup_record.get('flat_values', {})
    constraint_fields = list(template['constraint_fields'])
    explore_field = get_explore_field(template)

    option_buckets = []
    for field in constraint_fields:
        entries = flat_values.get(field, [])
        if not entries:
            return []
        literals = [build_literal_from_entry(field, e, G_FIELDS_CFG[field]) for e in entries]
        literals = dedupe_literals(literals)
        if not literals:
            return []
        option_buckets.append(literals)

    if explore_field is not None and not flat_values.get(explore_field, []):
        return []

    if not option_buckets:
        combos = [tuple()]
    else:
        from itertools import product
        combos = product(*option_buckets)

    candidates = []
    seen = set()
    for combo in combos:
        constraints = sorted(list(combo), key=literal_sort_key)
        binding_groups = infer_triggered_binding_groups(setup_record, constraints)
        candidate = {
            'domain': setup_record['domain'],
            'type': template['type'],
            'type_id': template['type_id'],
            'total': template['total'],
            'has_numeric': bool(template.get('has_numeric_field', False)),
            'has_negation': False,
            'constraints': constraints,
            'explore_field': explore_field,
            'constraint_count': template.get('constraint_count', len(constraints)),
            'binding_groups': binding_groups,
        }
        k = candidate_key(candidate)
        if k not in seen:
            seen.add(k)
            candidates.append(candidate)
    return candidates

def is_method_exploration(candidate):
    return candidate.get('type') == 'method_exploration'

def get_literal_matched_setup_ids(literal):
    global G_LITERAL_MATCH_CACHE
    lk = literal_key(literal)
    if lk in G_LITERAL_MATCH_CACHE:
        return G_LITERAL_MATCH_CACHE[lk]

    field = literal['field']
    if literal['op'] == '=' and 'parsed' not in literal and 'range' not in literal:
        matched = set(G_EXACT_VALUE_INDEX.get(field, {}).get(literal['value'], set()))
        G_LITERAL_MATCH_CACHE[lk] = matched
        return matched

    candidate_setups = G_FIELD_NONEMPTY_INDEX.get(field, set())
    matched = set()
    for idx in candidate_setups:
        rec = G_SETUPS[idx]
        entries = rec.get('flat_values', {}).get(field, [])
        if entries and any_entry_matches_literal(entries, literal):
            matched.add(idx)
    G_LITERAL_MATCH_CACHE[lk] = matched
    return matched


def get_template_eligible_setup_ids(template):
    fields = set(template['constraint_fields'])
    explore_field = get_explore_field(template)
    if explore_field is not None:
        fields.add(explore_field)
    eligible = None
    for f in fields:
        s = G_FIELD_NONEMPTY_INDEX.get(f, set())
        eligible = set(s) if eligible is None else eligible & s
        if not eligible:
            return set()
    return eligible if eligible is not None else set(range(len(G_SETUPS)))


def _as_entry_objects(raw_values):
    out = []
    for v in raw_values or []:
        if isinstance(v, dict):
            if 'value' in v:
                out.append(v)
        else:
            out.append({'value': v})
    return out


def binding_group_matches_setup(group, constraints, setup_record):
    items = (setup_record.get('item_bindings', {}) or {}).get(group['list_path'], [])
    if not items:
        return False
    relevant_fields = set(group['fields'])
    relevant_literals = [lit for lit in constraints if lit['field'] in relevant_fields]
    if len({lit['field'] for lit in relevant_literals}) < 2:
        return True
    for item in items:
        ok = True
        for lit in relevant_literals:
            entries = _as_entry_objects(item.get(lit['field'], []))
            if not entries or not any_entry_matches_literal(entries, lit):
                ok = False
                break
        if ok:
            return True
    return False


def candidate_passes_binding_checks(candidate, setup_record):
    groups = candidate.get('binding_groups', []) or []
    if not groups:
        return True
    return all(binding_group_matches_setup(group, candidate['constraints'], setup_record) for group in groups)


def fast_compute_support(candidate, eligible_setup_ids):
    constraints = candidate['constraints']
    if constraints:
        matched_sets = [get_literal_matched_setup_ids(lit) for lit in constraints]
        coarse_ids = set.intersection(*matched_sets) if matched_sets else set()
        coarse_ids &= eligible_setup_ids
    else:
        coarse_ids = set(eligible_setup_ids)

    cand = dict(candidate)
    if not coarse_ids:
        cand['support_paragraph_ids'] = []
        cand['support'] = 0
        cand['distinct_explored_values'] = [] if candidate.get('explore_field') else None
        cand['distinct_explored_value_count'] = 0 if candidate.get('explore_field') else None
        return cand

    paragraph_ids = set()
    explored_values = set()
    explore_field = candidate.get('explore_field')
    for idx in coarse_ids:
        rec = G_SETUPS[idx]
        if not candidate_passes_binding_checks(candidate, rec):
            continue
        paragraph_ids.add(str(rec['paragraph_id']))
        if explore_field is not None:
            for e in rec.get('flat_values', {}).get(explore_field, []):
                v = e.get('value')
                if v not in (None, ''):
                    explored_values.add(v)

    cand['support_paragraph_ids'] = sorted(paragraph_ids, key=lambda x: (not x.isdigit(), x))
    cand['support'] = len(paragraph_ids)
    if explore_field is not None:
        cand['distinct_explored_values'] = sorted(explored_values)
        cand['distinct_explored_value_count'] = len(explored_values)
    else:
        cand['distinct_explored_values'] = None
        cand['distinct_explored_value_count'] = None
    return cand


def process_one_template(template):
    eligible_setup_ids = get_template_eligible_setup_ids(template)
    if not eligible_setup_ids:
        return []
    unique_by_key = {}
    for idx in eligible_setup_ids:
        setup = G_SETUPS[idx]
        for cand in instantiate_candidates_from_setup(setup, template):
            k = candidate_key(cand)
            if k not in unique_by_key:
                unique_by_key[k] = cand
    return [fast_compute_support(cand, eligible_setup_ids) for cand in unique_by_key.values()]


def add_candidate_ids(candidates, domain, polarity='pos'):
    counters = Counter()
    for cand in candidates:
        tag = f'{cand["type_id"]}_{polarity}'
        counters[tag] += 1
        cand['candidate_id'] = f'{domain}_{tag}_{counters[tag]:06d}'


def compute_candidate_stats(candidates):
    stats = {
        'num_candidates': len(candidates),
        'support': {'avg': 0.0, 'min': None, 'max': None},
        'by_type': {},
        'by_type_id': {},
        'by_total': {},
        'by_has_numeric': {},
        'by_has_negation': {},
        'by_has_binding': {},
        'exploration': {
            'num_candidates': 0,
            'by_explore_field': {},
            'distinct_explored_value_count': {'avg': 0.0, 'min': None, 'max': None},
        },
    }
    if not candidates:
        return stats
    supports = [c['support'] for c in candidates]
    stats['support']['avg'] = round(sum(supports) / len(supports), 4)
    stats['support']['min'] = min(supports)
    stats['support']['max'] = max(supports)
    stats['by_type'] = dict(sorted(Counter(c['type'] for c in candidates).items()))
    stats['by_type_id'] = dict(sorted(Counter(c['type_id'] for c in candidates).items()))
    stats['by_total'] = dict(sorted(Counter(c['total'] for c in candidates).items()))
    stats['by_has_numeric'] = {str(k): v for k, v in sorted(Counter(c['has_numeric'] for c in candidates).items())}
    stats['by_has_negation'] = {str(k): v for k, v in sorted(Counter(c['has_negation'] for c in candidates).items())}
    stats['by_has_binding'] = {str(k): v for k, v in sorted(Counter(bool(c.get('binding_groups')) for c in candidates).items())}
    exploration_candidates = [c for c in candidates if c.get('explore_field') is not None]
    stats['exploration']['num_candidates'] = len(exploration_candidates)
    if exploration_candidates:
        stats['exploration']['by_explore_field'] = dict(sorted(Counter(c['explore_field'] for c in exploration_candidates).items()))
        dvals = [c['distinct_explored_value_count'] for c in exploration_candidates]
        stats['exploration']['distinct_explored_value_count']['avg'] = round(sum(dvals) / len(dvals), 4)
        stats['exploration']['distinct_explored_value_count']['min'] = min(dvals)
        stats['exploration']['distinct_explored_value_count']['max'] = max(dvals)
    return stats


def main(
    setup_records_path,
    field_template_combinations_path,
    field_registry_path,
    raw_output_path,
    filtered_output_path,
    raw_stats_path,
    filtered_stats_path,
    min_support=3,
    max_support=5,
    min_explore_values=3,
    min_method_explore_values=1,
    max_workers=None
):
    templates = load_template_combinations(field_template_combinations_path)
    if not templates:
        raise ValueError('No template combinations found.')
    first = templates[0]
    domain = first.get('domain') or 'unknown_domain'
    max_workers = max_workers or max(1, mp.cpu_count() - 1)

    all_candidates = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(setup_records_path, field_registry_path)
    ) as ex:
        futures = [ex.submit(process_one_template, tpl) for tpl in templates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing templates'):
            all_candidates.extend(fut.result())

    unique = {}
    for c in all_candidates:
        unique[candidate_key(c)] = c
    raw_candidates = list(unique.values())

    filtered_candidates = []
    for c in raw_candidates:
        if not (min_support <= c['support'] <= max_support):
            continue

        if c.get('explore_field') is not None:
            distinct_count = c.get('distinct_explored_value_count') or 0

            if is_method_exploration(c):
                if distinct_count < min_method_explore_values:
                    continue
            else:
                if distinct_count < min_explore_values:
                    continue

        filtered_candidates.append(c)

    add_candidate_ids(raw_candidates, domain, polarity='pos')
    add_candidate_ids(filtered_candidates, domain, polarity='pos')

    raw_stats = compute_candidate_stats(raw_candidates)
    filtered_stats = compute_candidate_stats(filtered_candidates)

    write_jsonl(raw_output_path, raw_candidates)
    write_jsonl(filtered_output_path, filtered_candidates)
    write_json(raw_stats_path, raw_stats)
    write_json(filtered_stats_path, filtered_stats)

    print(f'Saved raw candidates: {len(raw_candidates)} -> {raw_output_path}')
    print(
        f'Saved filtered candidates '
        f'(support {min_support}-{max_support}, '
        f'min_explore_values={min_explore_values}, '
        f'min_method_explore_values={min_method_explore_values}): '
        f'{len(filtered_candidates)} -> {filtered_output_path}'
    )
    print(f'Saved raw stats -> {raw_stats_path}')
    print(f'Saved filtered stats -> {filtered_stats_path}')




if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        # "cell_culture",
        "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials"
    ]

    max_workers = min(32, os.cpu_count() or 1, 64)

    for d in domains:
        setup_records_path = f"seed_data/{d}/constraints/setup_records.jsonl"

        field_template_combinations_path = f"seed_data/{d}/field_template_combinations.json"
        field_registry_path = f"seed_data/{d}/field_registry.json"

        raw_output_path = f"seed_data/{d}/constraints/positive_constraint_candidates_raw.jsonl"
        filtered_output_path = f"seed_data/{d}/constraints/constraint_candidates_positive.jsonl"

        raw_stats_path = f"seed_data/{d}/constraints/positive_constraint_candidates_raw.stats.json"
        filtered_stats_path = f"seed_data/{d}/constraints/constraint_candidates_positive.stats.json"

        print(f"\n=== Processing domain: {d} ===")
        main(
            setup_records_path=setup_records_path,
            field_template_combinations_path=field_template_combinations_path,
            field_registry_path=field_registry_path,
            raw_output_path=raw_output_path,
            filtered_output_path=filtered_output_path,
            raw_stats_path=raw_stats_path,
            filtered_stats_path=filtered_stats_path,
            min_support=3,
            max_support=8,
            min_explore_values=3,
            min_method_explore_values=1,
            max_workers=max_workers,
        )


# python scripts/constraint_set/build_positive_candidates.py

