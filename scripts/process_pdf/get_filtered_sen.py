# import json

# # tp = 'electrochemistry'
# tp = 'nanoparticle'
# input_file = f'seed_data/{tp}/cited_sentence.json'
# output_file = f'seed_data/{tp}/filtered_cited_sentence.json'
# input_txt = f'seed_data/{tp}/gold_doi.txt' 
# output_txt = f'seed_data/{tp}/appendix_doi.txt' 

# possitive_words = [
#     'avoid', 'prevent','exclude','order to', 'due to',
#     'study', 'studies', 'work', 'describe', 'reported', 'according', 'following', 'previous', 'performed'
# ]

# # def has_any_real_doi(rec: dict) -> bool:
# #     for cref in rec.get("cited_refs", []):
# #         for doi in (cref.get("dois") or []):
# #             if doi is None:
# #                 continue
# #             s = str(doi).strip()
# #             if s and s.lower() != "none":
# #                 return True
# #     return False

# def has_any_real_doi(rec: dict) -> bool:
#     for cref in rec.get("cited_refs", []):
#         for doi in (cref.get("dois") or []):
#             if doi is None:
#                 return False
#     return True


# with open(input_file, 'r', encoding='utf-8') as f:
#     recs = json.load(f)

# with open(input_txt, 'r', encoding='utf-8') as f:
#     input_dois = [line.strip() for line in f if line.strip()]

# filtered_recs = []
# output_dois = []
# for rec in recs:
#     if any([w in rec['sentence'] for w in possitive_words]) and 'DFT ' not in rec['sentence'] and has_any_real_doi(rec):
#         filtered_recs.append(rec)
#         for cref in rec.get("cited_refs", []):
#             for doi in (cref.get("dois") or []):
#                 if doi not in input_dois:
#                     output_dois.append(doi)

# output_dois = sorted(set(output_dois))
# with open(output_txt, 'w', encoding='utf-8') as f:
#     f.write('\n'.join(output_dois))
# print(f'需要下载的文献数量：{len(output_dois)}')

# print(f"过滤后的文献数量：{len({rec['doi'] for rec in filtered_recs})}")
# print(len(filtered_recs),len(recs))
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(filtered_recs, f, ensure_ascii=False, indent=2)


# python scripts/process_pdf/get_filtered_sen.py


import json

tp = 'electrochemistry'
# tp = 'nanoparticle'
input_file = f'seed_data/{tp}/cited_sentence.json'
output_file = f'seed_data/{tp}/filtered_cited_sentence.json'
input_txt = f'seed_data/{tp}/gold_doi.txt' 
output_txt = f'seed_data/{tp}/appendix_doi_valid.txt' 

valid_doi_path = f'seed_data/{tp}/{tp}_s2orc_or_oa_available.txt'
with open(valid_doi_path, 'r', encoding='utf-8') as f:
    valid_dois = [line.strip() for line in f if line.strip()]


possitive_words = [
    'avoid', 'prevent','exclude','order to', 'due to',
    'study', 'studies', 'work', 'describe', 'reported', 'according', 'following', 'previous', 'performed'
]

# def has_any_real_doi(rec: dict) -> bool:
#     for cref in rec.get("cited_refs", []):
#         for doi in (cref.get("dois") or []):
#             if doi is None:
#                 continue
#             s = str(doi).strip()
#             if s and s.lower() != "none":
#                 return True
#     return False

def has_any_real_doi(rec: dict) -> bool:
    for cref in rec.get("cited_refs", []):
        for doi in (cref.get("dois") or []):
            if doi is None:
                return False
            if doi not in valid_dois:
                return False
    return True


with open(input_file, 'r', encoding='utf-8') as f:
    recs = json.load(f)

with open(input_txt, 'r', encoding='utf-8') as f:
    input_dois = [line.strip() for line in f if line.strip()]

filtered_recs = []
output_dois = []
for rec in recs:
    if any([w in rec['sentence'] for w in possitive_words]) and 'DFT ' not in rec['sentence'] and has_any_real_doi(rec):
        filtered_recs.append(rec)
        for cref in rec.get("cited_refs", []):
            for doi in (cref.get("dois") or []):
                if doi not in input_dois:
                    output_dois.append(doi)

output_dois = sorted(set(output_dois))
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_dois))
print(f'需要下载的文献数量：{len(output_dois)}')

print(f"过滤后的文献数量：{len({rec['doi'] for rec in filtered_recs})}")
print(len(filtered_recs),len(recs))
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_recs, f, ensure_ascii=False, indent=2)

