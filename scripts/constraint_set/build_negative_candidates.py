import json
import os
import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

EPS = 1e-8

# =========================
# IO Utils
# =========================

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

# =========================
# Globals & Helper Functions
# =========================

G_SETUPS = None
G_FIELDS_CFG = None
G_FIELD_NONEMPTY_INDEX = None
G_EXACT_VALUE_INDEX = None
G_VALUE_FREQ = None
G_LITERAL_MATCH_CACHE = None
G_TEMPLATE_MAP = None


def normalize_unit(unit):
    if unit is None:
        return None
    unit = str(unit).strip()
    if not unit:
        return None
    return unit.replace('µ', 'μ').replace('ug', 'μg').replace('mcg', 'μg')


def literal_key(lit):
    payload = {'field': lit['field'], 'op': lit['op']}
    if lit['op'] in ('=', '!='):
        payload['value'] = lit['value']
        if 'parsed' in lit:
            payload['parsed'] = lit['parsed']
    elif lit['op'] == 'between':
        payload['range'] = lit['range']
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def candidate_key(candidate):
    payload = {
        'domain': candidate['domain'],
        'type': candidate['type'],
        'type_id': candidate['type_id'],
        'constraints': sorted(
            candidate['constraints'],
            key=lambda x: (x['field'], x['op'], str(x.get('value', '')), json.dumps(x, ensure_ascii=False, sort_keys=True)),
        ),
        'explore_field': candidate.get('explore_field'),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def init_worker(setup_records_path, field_registry_path, template_path):
    global G_SETUPS, G_FIELDS_CFG, G_FIELD_NONEMPTY_INDEX, G_EXACT_VALUE_INDEX
    global G_VALUE_FREQ, G_LITERAL_MATCH_CACHE, G_TEMPLATE_MAP

    G_SETUPS = list(read_jsonl(setup_records_path))
    reg_data = read_json(field_registry_path)
    G_FIELDS_CFG = reg_data.get('fields', reg_data)

    G_FIELD_NONEMPTY_INDEX = defaultdict(set)
    G_EXACT_VALUE_INDEX = defaultdict(lambda: defaultdict(set))
    G_VALUE_FREQ = defaultdict(Counter)

    for i, rec in enumerate(G_SETUPS):
        flat = rec.get('flat_values', {}) or {}
        for f, entries in flat.items():
            if entries:
                G_FIELD_NONEMPTY_INDEX[f].add(i)
            for e in entries:
                if isinstance(e, dict) and 'value' in e:
                    v = e['value']
                    G_EXACT_VALUE_INDEX[f][v].add(i)
                    G_VALUE_FREQ[f][v] += 1

    G_LITERAL_MATCH_CACHE = {}
    templates = load_template_combinations(template_path)
    G_TEMPLATE_MAP = {}
    for t in templates:
        key = json.dumps({'type_id': t['type_id'], 'fields': sorted(t['constraint_fields'])}, sort_keys=True)
        G_TEMPLATE_MAP[key] = t

# =========================
# Matching Logic
# =========================


def eq_number(a, b):
    return abs(float(a) - float(b)) <= EPS


def numeric_match(literal, entry):
    s_parsed = entry.get('parsed')
    if not s_parsed:
        return False
    if literal['op'] == '=':
        q_parsed = literal['parsed']
        if normalize_unit(q_parsed.get('unit')) != normalize_unit(s_parsed.get('unit')):
            return False
        if s_parsed['kind'] == 'point':
            return eq_number(s_parsed['number'], q_parsed['number'])
        if s_parsed['kind'] == 'range':
            return s_parsed['lower'] - EPS <= q_parsed['number'] <= s_parsed['upper'] + EPS
    elif literal['op'] == 'between':
        ql, qr = literal['range']['lower'], literal['range']['upper']
        if normalize_unit(literal['range'].get('unit')) != normalize_unit(s_parsed.get('unit')):
            return False
        if s_parsed['kind'] == 'point':
            return ql - EPS <= s_parsed['number'] <= qr + EPS
        if s_parsed['kind'] == 'range':
            return ql - EPS <= s_parsed['lower'] and s_parsed['upper'] <= qr + EPS
    return False


def literal_matches_entries(literal, entries):
    if not entries:
        return False
    if literal['op'] == '!=':
        return all(e.get('value') != literal['value'] for e in entries)
    for e in entries:
        if literal['op'] == '=':
            if 'parsed' in literal:
                if numeric_match(literal, e):
                    return True
            elif e.get('value') == literal['value']:
                return True
        elif literal['op'] == 'between':
            if numeric_match(literal, e):
                return True
    return False


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
            if not entries or not literal_matches_entries(lit, entries):
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


def get_matched_ids(literal):
    lk = literal_key(literal)
    if lk in G_LITERAL_MATCH_CACHE:
        return G_LITERAL_MATCH_CACHE[lk]

    field = literal['field']
    if literal['op'] == '=' and 'parsed' not in literal and 'range' not in literal:
        res = set(G_EXACT_VALUE_INDEX.get(field, {}).get(literal['value'], set()))
    else:
        candidate_ids = G_FIELD_NONEMPTY_INDEX.get(field, set())
        res = {
            i for i in candidate_ids
            if literal_matches_entries(literal, G_SETUPS[i].get('flat_values', {}).get(field, []))
        }

    G_LITERAL_MATCH_CACHE[lk] = res
    return res


def fast_compute_support(candidate):
    matched_sets = [get_matched_ids(lit) for lit in candidate['constraints']]
    if not matched_sets:
        return 0, [], []

    common_ids = set.intersection(*matched_sets)
    final_pids = set()
    explored_vals = set()
    ex_field = candidate.get('explore_field')

    for idx in common_ids:
        rec = G_SETUPS[idx]
        if not candidate_passes_binding_checks(candidate, rec):
            continue

        if ex_field:
            ex_entries = rec.get('flat_values', {}).get(ex_field, [])
            has_explore_value = any(e.get('value') not in (None, '') for e in ex_entries)
            if not has_explore_value:
                continue

            for e in ex_entries:
                if e.get('value') not in (None, ''):
                    explored_vals.add(e['value'])

        final_pids.add(str(rec['paragraph_id']))

    return len(final_pids), sorted(list(final_pids)), sorted(list(explored_vals))


def get_top_negation_values(field, base_value, min_negated_value_freq, max_negation_values_per_field):
    freq_counter = G_VALUE_FREQ.get(field, Counter())
    eligible = [
        (v, freq)
        for v, freq in freq_counter.items()
        if v != base_value and freq >= min_negated_value_freq
    ]
    eligible.sort(key=lambda x: (-x[1], str(x[0])))
    return [v for v, _ in eligible[:max_negation_values_per_field]]


def process_chunk(chunk, min_negated_value_freq, max_negation_values_per_field):
    results = []
    for base in chunk:
        tpl_key = json.dumps({'type_id': base['type_id'], 'fields': sorted([l['field'] for l in base['constraints']])}, sort_keys=True)
        template = G_TEMPLATE_MAP.get(tpl_key)
        if not template:
            continue

        neg_fields = template.get('negatable_fields_in_combo', [])
        for f in neg_fields:
            base_lit = next((l for l in base['constraints'] if l['field'] == f and l['op'] == '=' and 'parsed' not in l), None)
            if not base_lit:
                continue
            base_val = base_lit['value']

            alt_values = get_top_negation_values(
                field=f,
                base_value=base_val,
                min_negated_value_freq=min_negated_value_freq,
                max_negation_values_per_field=max_negation_values_per_field,
            )
            for v in alt_values:
                new_constraints = []
                for lit in base['constraints']:
                    if lit['field'] == f:
                        new_constraints.append({'field': f, 'op': '!=', 'value': v})
                    else:
                        new_constraints.append(lit)
                neg_cand = {
                    'domain': base['domain'],
                    'type': base['type'],
                    'type_id': base['type_id'],
                    'total': base['total'],
                    'constraints': new_constraints,
                    'explore_field': base.get('explore_field'),
                    'has_negation': True,
                    'binding_groups': base.get('binding_groups', []),
                    'source_base_id': base.get('candidate_id'),
                    'negated_field': f,
                    'negated_value': v,
                    'negated_value_frequency': G_VALUE_FREQ.get(f, Counter()).get(v, 0),
                }
                supp, pids, ex_vals = fast_compute_support(neg_cand)
                if supp > 0:
                    neg_cand.update({
                        'support': supp,
                        'support_paragraph_ids': pids,
                        'distinct_explored_values': ex_vals,
                        'distinct_explored_value_count': len(ex_vals),
                    })
                    results.append(neg_cand)
    return results

def is_method_exploration(candidate):
    return candidate.get('type') == 'method_exploration'


def main(
    setup_records_path,
    positive_candidates_path,
    field_template_combinations_path,
    field_registry_path,
    filtered_output_path,
    raw_stats_path=None,
    filtered_stats_path=None,
    base_min_support=3,
    base_max_support=20,
    min_support=3,
    max_support=5,
    min_explore_values=3,
    min_method_explore_values=1,  
    min_negated_value_freq=5,
    max_negation_values_per_field=10,
    max_workers=None,
    chunk_size=50,
):
    base_candidates = [
        c for c in read_jsonl(positive_candidates_path)
        if base_min_support <= c.get('support', 0) <= base_max_support
    ]
    if not base_candidates:
        print(f'[WARN] No base candidates found in support range [{base_min_support}, {base_max_support}]')
        return

    domain = base_candidates[0]['domain']
    max_workers = max_workers or max(1, mp.cpu_count() - 1)
    chunks = [base_candidates[i:i + chunk_size] for i in range(0, len(base_candidates), chunk_size)]

    raw_neg_candidates = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(setup_records_path, field_registry_path, field_template_combinations_path),
    ) as ex:
        futures = [
            ex.submit(process_chunk, chunk, min_negated_value_freq, max_negation_values_per_field)
            for chunk in chunks
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f'Generating Negations [{domain}]'):
            raw_neg_candidates.extend(fut.result())

    unique_by_key = {}
    for cand in raw_neg_candidates:
        if not (min_support <= cand['support'] <= max_support):
            continue
        if cand.get('explore_field'):
            distinct_count = cand.get('distinct_explored_value_count', 0)

            if is_method_exploration(cand):
                if distinct_count < min_method_explore_values:
                    continue
            else:
                if distinct_count < min_explore_values:
                    continue
        
        k = candidate_key(cand)
        if k not in unique_by_key:
            unique_by_key[k] = cand

    final_list = list(unique_by_key.values())
    for i, c in enumerate(final_list):
        c['candidate_id'] = f"{domain}_{c['type_id']}_neg_{i+1:06d}"

    write_jsonl(filtered_output_path, final_list)

    if raw_stats_path:
        raw_stats = {
            'domain': domain,
            'base_candidate_count': len(base_candidates),
            'raw_negative_candidate_count': len(raw_neg_candidates),
            'base_support_range': [base_min_support, base_max_support],
            'min_negated_value_freq': min_negated_value_freq,
            'max_negation_values_per_field': max_negation_values_per_field,
        }
        write_json(raw_stats_path, raw_stats)

    if filtered_stats_path:
        filtered_stats = {
            'domain': domain,
            'final_negative_candidate_count': len(final_list),
            'filtered_support_range': [min_support, max_support],
            'min_explore_values': min_explore_values,
            'min_method_explore_values': min_method_explore_values,
            'min_negated_value_freq': min_negated_value_freq,
            'by_type_id': dict(sorted(Counter(c['type_id'] for c in final_list).items())),
            'by_negated_field': dict(sorted(Counter(c['negated_field'] for c in final_list).items())),
        }
        write_json(filtered_stats_path, filtered_stats)

    print(f'[OK] Processed {len(base_candidates)} base candidates.')
    print(f'[OK] Saved {len(final_list)} filtered negative candidates to {filtered_output_path}')


if __name__ == '__main__':
    domains = [
        # 'electrochemistry',
        # 'cell_culture',
        'immunotechnology',
        # 'inorganic_materials',
        # 'organic_synthesis',
        # 'polymer_materials',
    ]

    max_cpus = os.cpu_count() or 1

    for d in domains:
        print(f'\n=== Processing Negative Domain: {d} ===')
        main(
            setup_records_path=f'seed_data/{d}/constraints/setup_records.jsonl',
            positive_candidates_path=f'seed_data/{d}/constraints/constraint_candidates_positive.jsonl',
            field_template_combinations_path=f'seed_data/{d}/field_template_combinations.json',
            field_registry_path=f'seed_data/{d}/field_registry.json',

            filtered_output_path=f'seed_data/{d}/constraints/constraint_candidates_negative.jsonl',
            raw_stats_path=f'seed_data/{d}/constraints/negative_constraint_candidates_raw.stats.json',
            filtered_stats_path=f'seed_data/{d}/constraints/constraint_candidates_negative.stats.json',
            base_min_support=3,
            base_max_support=40,
            min_support=3,
            max_support=8,
            min_explore_values=3,
            min_method_explore_values=1,
            min_negated_value_freq=5,
            max_negation_values_per_field=10,  # 最多频率最高的前k个取值进行判定
            max_workers=max_cpus,
            chunk_size=50,
        )


# python scripts/constraint_set/build_negative_candidates.py
