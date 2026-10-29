import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {e}') from e


def write_jsonl(records: Iterable[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ''
    s = unicodedata.normalize('NFKC', str(text))
    s = s.replace('\u00a0', ' ')
    s = s.replace('µ', 'μ')
    s = s.replace('−', '-')
    s = s.replace('–', '-')
    s = s.replace('—', '-')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def load_custom_unit_aliases(path: Optional[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not path:
        return alias_map
    p = Path(path)
    if not p.exists():
        return alias_map

    with p.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.split('#', 1)[0].strip()
            if not line or '=' not in line:
                continue
            parts = [x.strip() for x in line.split('=')]
            if not parts or not parts[0]:
                continue
            canonical = parts[0]
            candidates = [canonical]
            for extra in parts[2:]:
                if extra and extra != '_':
                    candidates.append(extra)
            canonical_norm = normalize_text(canonical).lower()
            for cand in candidates:
                alias_map[normalize_text(cand).lower()] = canonical_norm
    return alias_map


def normalize_unit_text(
    unit_text: Optional[str],
    field_name: Optional[str] = None,
    unit_hint: Optional[str] = None,
    custom_unit_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if unit_text is None:
        unit_text = unit_hint
    if unit_text is None:
        return None

    s = normalize_text(unit_text)
    if not s:
        s = normalize_text(unit_hint)
        if not s:
            return None

    s = s.lower()
    s = s.replace('μ', 'u')
    s = s.replace('²', '2').replace('³', '3')
    s = re.sub(r'\bper\b', '/', s)
    s = re.sub(r'\s*/\s*', '/', s)
    s = re.sub(r'\s+', ' ', s).strip()

    generic_map = {
        '%': '%', 'percent': '%', 'percentage': '%',
        'hr': 'h', 'hrs': 'h', 'hour': 'h', 'hours': 'h', 'h': 'h',
        'min': 'min', 'mins': 'min', 'minute': 'min', 'minutes': 'min',
        'd': 'd', 'day': 'd', 'days': 'd',
        'c': 'c', '°c': 'c',
    }
    if s in generic_map:
        return generic_map[s]

    if field_name == 'seeding_density':
        s = s.replace('cell/', 'cells/')
        s = s.replace('cell ', 'cells ')
        patterns = [
            (r'^cells?/cm\^?\s*2$', 'cells/cm2'),
            (r'^cells?/mm\^?\s*2$', 'cells/mm2'),
            (r'^cells?/m\^?\s*2$', 'cells/m2'),
            (r'^cells?/100\s*u?m\^?\s*2$', 'cells/100um2'),
            (r'^cells?/ml$', 'cells/ml'),
            (r'^cells?/l$', 'cells/l'),
            (r'^cells?/wells?$', 'cells/well'),
            (r'^cells?/dishes?$', 'cells/dish'),
            (r'^cells?/flasks?$', 'cells/flask'),
            (r'^cells?/slides?$', 'cells/slide'),
            (r'^cells?/scaffolds?$', 'cells/scaffold'),
            (r'^cells?/patches?$', 'cells/patch'),
            (r'^cells?$', 'cells'),
        ]
        s_compact = re.sub(r'\s+', '', s) if 'cells/' in s else s
        for pattern, canonical in patterns:
            if re.fullmatch(pattern, s_compact):
                return canonical
        return s_compact

    if custom_unit_aliases:
        exact = custom_unit_aliases.get(s)
        if exact:
            return exact
    return s


SCI_TOKEN_RE = r'[+-]?\d+(?:\.\d+)?\s*[x×*]\s*10(?:\s*\^\s*|\s+)[+-]?\d+'
SIMPLE_TOKEN_RE = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
NUMBER_TOKEN_RE = rf'(?:{SCI_TOKEN_RE}|{SIMPLE_TOKEN_RE})'
RANGE_SEP_RE = r'(?:-|to|~)'


def parse_number_token(token: str) -> Optional[float]:
    s = normalize_text(token)
    if not s:
        return None
    s = s.replace('×', 'x').replace('*', 'x').replace(',', '')
    s = re.sub(r'\s+', ' ', s).strip()
    m_sci = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*x\s*10(?:\s*\^\s*|\s+)([+-]?\d+)', s, flags=re.IGNORECASE)
    if m_sci:
        return float(m_sci.group(1)) * (10 ** int(m_sci.group(2)))
    try:
        return float(s)
    except ValueError:
        return None


def parse_numeric_value(
    raw_value: Optional[str],
    field_name: Optional[str] = None,
    unit_hint: Optional[str] = None,
    custom_unit_aliases: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    if raw_value is None:
        return None
    s = normalize_text(raw_value)
    if not s:
        return None

    m_ratio = re.fullmatch(rf'\s*({SIMPLE_TOKEN_RE})\s*[:/]\s*({SIMPLE_TOKEN_RE})\s*', s, flags=re.IGNORECASE)
    if m_ratio:
        num1 = parse_number_token(m_ratio.group(1))
        num2 = parse_number_token(m_ratio.group(2))
        if num1 is not None and num2 is not None:
            return {
                'kind': 'ratio',
                'number': num2,
                'unit': normalize_unit_text(None, field_name, unit_hint, custom_unit_aliases),
                'raw': s,
                'numerator': num1,
                'denominator': num2,
            }

    m_range = re.fullmatch(
        rf'\s*({NUMBER_TOKEN_RE})\s*{RANGE_SEP_RE}\s*({NUMBER_TOKEN_RE})(?:\s*(.*?))?\s*$',
        s,
        flags=re.IGNORECASE,
    )
    if m_range:
        lower = parse_number_token(m_range.group(1))
        upper = parse_number_token(m_range.group(2))
        if lower is not None and upper is not None:
            unit = normalize_unit_text(m_range.group(3), field_name, unit_hint, custom_unit_aliases)
            return {'kind': 'range', 'lower': lower, 'upper': upper, 'unit': unit, 'raw': s}

    m_point = re.search(NUMBER_TOKEN_RE, s, flags=re.IGNORECASE)
    if m_point:
        number = parse_number_token(m_point.group(0))
        if number is not None:
            remainder = s[m_point.end():].strip()
            unit = normalize_unit_text(remainder, field_name, unit_hint, custom_unit_aliases)
            return {'kind': 'point', 'number': number, 'unit': unit, 'raw': s}

    return None


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        s = normalize_text(v)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def collect_leaf_values_from_setup(
    node: Any,
    registry_fields: set,
    path_parts: Optional[List[str]] = None,
    out: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    if path_parts is None:
        path_parts = []
    if out is None:
        out = defaultdict(list)

    if isinstance(node, dict):
        if 'value' in node:
            field_name = '.'.join(path_parts)
            if field_name in registry_fields:
                raw_values = node.get('value', [])
                if not isinstance(raw_values, list):
                    raw_values = [raw_values]
                for raw in raw_values:
                    if raw is not None:
                        out[field_name].append(str(raw))
            return out
        for key, value in node.items():
            if key == 'evidence':
                continue
            collect_leaf_values_from_setup(value, registry_fields, path_parts + [key], out)
        return out

    if isinstance(node, list):
        for item in node:
            collect_leaf_values_from_setup(item, registry_fields, path_parts, out)
    return out


def load_binding_group_rules(path: Optional[str]) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = load_json(str(p))
    groups = data.get('binding_groups', []) if isinstance(data, dict) else data
    out = []
    for g in groups:
        list_path = g.get('list_path') or g.get('group_name')
        fields = list(g.get('fields', []))
        if not list_path or not fields:
            continue
        out.append({
            'group_name': g.get('group_name', list_path),
            'list_path': list_path,
            'fields': fields,
        })
    return out


def collect_nodes_at_path(node: Any, path_parts: List[str]) -> List[Any]:
    if not path_parts:
        return [node]
    part = path_parts[0]
    rest = path_parts[1:]
    out = []
    if isinstance(node, list):
        for item in node:
            out.extend(collect_nodes_at_path(item, path_parts))
        return out
    if isinstance(node, dict) and part in node:
        return collect_nodes_at_path(node[part], rest)
    return out


def build_entry_objects(raw_values: List[str], field_name: str, field_registry: Dict[str, dict], custom_unit_aliases: Dict[str, str]) -> List[dict]:
    raw_values = unique_preserve_order(raw_values)
    if not raw_values:
        return []
    meta = field_registry[field_name]
    value_type = meta.get('value_type')
    unit_hint = meta.get('unit')
    items: List[dict] = []
    for raw_value in raw_values:
        obj = {'value': raw_value}
        if value_type == 'numeric':
            obj['parsed'] = parse_numeric_value(
                raw_value,
                field_name=field_name,
                unit_hint=unit_hint,
                custom_unit_aliases=custom_unit_aliases,
            )
        items.append(obj)
    return items


def collect_item_bindings_from_setup(
    setup: dict,
    binding_groups: List[dict],
    field_registry: Dict[str, dict],
    custom_unit_aliases: Dict[str, str],
) -> Dict[str, List[dict]]:
    item_bindings: Dict[str, List[dict]] = {}
    for group in binding_groups:
        list_path = group['list_path']
        fields = set(group['fields'])
        nodes = collect_nodes_at_path(setup, list_path.split('.'))
        items_out: List[dict] = []
        for node in nodes:
            raw_flat = collect_leaf_values_from_setup(node, fields, path_parts=list_path.split('.'))
            item_obj = {}
            for field_name, raw_values in raw_flat.items():
                entries = build_entry_objects(raw_values, field_name, field_registry, custom_unit_aliases)
                if entries:
                    item_obj[field_name] = entries
            if item_obj:
                items_out.append(item_obj)
        if items_out:
            item_bindings[list_path] = items_out
    return item_bindings


def init_field_stats(field_registry: Dict[str, dict]) -> Dict[str, dict]:
    stats = {}
    for field_name, meta in field_registry.items():
        item = {
            'bucket': meta.get('bucket'),
            'value_type': meta.get('value_type'),
            'setup_count': 0,
            'total_values': 0,
        }
        if meta.get('value_type') == 'numeric':
            item.update({
                'parseable_values': 0,
                'unparseable_values': 0,
                'point_values': 0,
                'range_values': 0,
                'ratio_values': 0,
            })
        stats[field_name] = item
    return stats


def transform_evidence_to_setup_records(
    domain: str,
    evidence_path: str,
    field_registry: Dict[str, dict],
    custom_units_path: Optional[str] = None,
    binding_group_rules_path: Optional[str] = None,
) -> Tuple[List[dict], dict]:
    registry_fields = set(field_registry.keys())
    custom_unit_aliases = load_custom_unit_aliases(custom_units_path)
    binding_groups = load_binding_group_rules(binding_group_rules_path)

    records: List[dict] = []
    field_stats = init_field_stats(field_registry)

    total_paragraphs = 0
    valid_paragraphs = 0
    invalid_paragraphs = 0
    total_setups = 0
    setup_records_with_item_bindings = 0
    item_binding_group_counts = defaultdict(int)

    for entry in iter_jsonl(evidence_path):
        total_paragraphs += 1
        result = entry.get('result', {})
        if not result.get('is_valid', False):
            invalid_paragraphs += 1
            continue
        valid_paragraphs += 1
        paragraph_id = str(entry.get('id'))
        setups = result.get('setups', []) or []

        for setup in setups:
            total_setups += 1
            setup_id = setup.get('setup_id')
            raw_flat = collect_leaf_values_from_setup(setup, registry_fields)
            sparse_flat_values: Dict[str, List[dict]] = {}
            for field_name, raw_values in raw_flat.items():
                items = build_entry_objects(raw_values, field_name, field_registry, custom_unit_aliases)
                if items:
                    sparse_flat_values[field_name] = items
            if not sparse_flat_values:
                continue

            item_bindings = collect_item_bindings_from_setup(
                setup,
                binding_groups,
                field_registry,
                custom_unit_aliases,
            )
            if item_bindings:
                setup_records_with_item_bindings += 1
                for k, v in item_bindings.items():
                    if v:
                        item_binding_group_counts[k] += 1

            records.append({
                'domain': domain,
                'paragraph_id': paragraph_id,
                'setup_id': setup_id,
                'flat_values': sparse_flat_values,
                'item_bindings': item_bindings,
            })

            for field_name, items in sparse_flat_values.items():
                fs = field_stats[field_name]
                fs['setup_count'] += 1
                fs['total_values'] += len(items)
                if fs['value_type'] == 'numeric':
                    for item in items:
                        parsed = item.get('parsed')
                        if parsed is None:
                            fs['unparseable_values'] += 1
                            continue
                        fs['parseable_values'] += 1
                        kind = parsed.get('kind')
                        if kind == 'point':
                            fs['point_values'] += 1
                        elif kind == 'range':
                            fs['range_values'] += 1
                        elif kind == 'ratio':
                            fs['ratio_values'] += 1

    summary = {
        'domain': domain,
        'total_paragraphs': total_paragraphs,
        'valid_paragraphs': valid_paragraphs,
        'invalid_paragraphs': invalid_paragraphs,
        'total_setup_records': len(records),
        'setup_records_with_item_bindings': setup_records_with_item_bindings,
        'item_binding_group_counts': dict(sorted(item_binding_group_counts.items())),
        'field_stats': field_stats,
    }
    return records, summary



if __name__ == '__main__':
    domains = [
        "electrochemistry",
        # "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        # "polymer_materials"
    ]

    for domain in domains:
        registry_path = f'seed_data/{domain}/field_registry.json'
        evidence_path = f'seed_data/{domain}/llm_cleaning/evidence_span.jsonl'
        setup_records_path = f'seed_data/{domain}/constraints/setup_records.jsonl'
        stats_path = f'seed_data/{domain}/constraints/setup_records.stats.json'
        custom_units_path = 'seed_data/scripts/constraint_set/custom_units.txt'
        binding_group_rules_path = f'seed_data/{domain}/binding_group_rules.json'

        field_registry = load_json(registry_path)
        records, stats = transform_evidence_to_setup_records(
            domain=domain,
            evidence_path=evidence_path,
            field_registry=field_registry,
            custom_units_path=custom_units_path,
            binding_group_rules_path=binding_group_rules_path,
        )
        write_jsonl(records, setup_records_path)
        save_json(stats, stats_path)
        print(f'Saved {len(records)} setup records to: {setup_records_path}')
        print(f'Saved stats to: {stats_path}')

# python scripts/constraint_set/get_setup_records.py


