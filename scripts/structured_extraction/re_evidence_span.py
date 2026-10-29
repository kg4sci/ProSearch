import re
import json
from pathlib import Path
from collections import Counter


class BaseLocator:
    def __init__(self):
        self.rules = {}

    def locate_all(self, text: str):
        evidences = {}
        for field, pattern in self.rules.items():
            matches = []  # 存储所有匹配
            for m in pattern.finditer(text):  # 使用 finditer 获取所有匹配
                matches.append({
                    "span": m.group(0),
                    "context": text[max(0, m.start()-40): min(len(text), m.end()+40)],
                })
            if matches:
                evidences[field] = matches  # 保存所有匹配结果
        return evidences


# =========================
# 1) electrochemistry
# =========================
class ElectrochemistryLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        conc_pat = r"(?:\d+(?:\.\d+)?)\s*(?:mM|M|mol\s*/\s*L|mol\s*L[-–]1)\b"

        electrolyte_alt = (
            r"(?:KOH|NaOH|LiOH|H2SO4|HClO4|HCl|HNO3|KHCO3|NaHCO3|K2CO3|Na2CO3|"
            r"K2SO4|Na2SO4|KCl|NaCl|LiClO4|NaClO4|KClO4|TBAPF6|TEABF4|TBAClO4|"
            r"LiPF6|LiTFSI|PBS|phosphate\s+buffer|borate\s+buffer)"
        )

        conc_plus_elec = rf"(?:{conc_pat})\s*(?:{electrolyte_alt})\b"

        self.rules = {
            "task": re.compile(
                r"\b(HER|OER|ORR|CO2RR|NRR|electrocataly[st]|water\s+splitting|reduction\s+of|oxidation\s+of|testing\s+of|performance\s+for)\b",
                re.IGNORECASE
            ),
            "electrolyte_chemical": re.compile(conc_plus_elec, re.IGNORECASE),
            "electrolyte_concentration": re.compile(conc_pat, re.IGNORECASE),
            "reference_electrode": re.compile(
                r"\b(Ag/AgCl|SCE|saturated\s+calomel|Hg/HgO|Hg/HgSO4|RHE|SHE|NHE)\b",
                re.IGNORECASE
            ),
            "working_electrode_material": re.compile(
                r"\b(glassy\s+carbon(?:\s+electrode)?|GCE|carbon\s+paper|carbon\s+cloth|graphite(?:\s+rod)?|"
                r"nickel\s+foam|Ni\s+foam|Pt(?:\s+foil|\s+wire|\s+electrode)?|Au(?:\s+foil|\s+electrode)?|"
                r"Cu(?:\s+foil|\s+electrode)?|FTO|ITO|Ti(?:\s+foil)?|stainless\s+steel)\b",
                re.IGNORECASE
            ),
            "pH": re.compile(r"\bpH\s*[:=~≈]?\s*\d+(?:\.\d+)?\b", re.IGNORECASE),
            "temperature": re.compile(
                r"(?:(?:-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|K)\b|(?:room|ambient)\s+temperature)",
                re.IGNORECASE
            ),
            "atmosphere": re.compile(
                r"\b((?:Ar|N2|O2|CO2|air)\s*(?:saturated|purged|bubbled|flow|blanketed|atmosphere)|"
                r"under\s+(?:Ar|N2|O2|CO2|vacuum|ambient)|purged\s+with\s+(?:Ar|N2|O2|CO2)|"
                r"continuous\s+flow\s+of)\b",
                re.IGNORECASE
            ),
            "potential_scale_reported": re.compile(
                r"\b(?:vs\.?|versus|referenced\s+to|referenced\s+vs\.?|calibrated\s+(?:to|against|vs\.?)|reported\s+(?:to|vs\.?))\s*(?:RHE|SHE|NHE|SCE|Ag/AgCl|Hg/HgO)\b",
                re.IGNORECASE
            ),
            "RHE_conversion_statement": re.compile(
                r"\b("
                r"converted\s+(?:to|into)\s+the?\s*RHE(?:\s+scale)?|"
                r"all\s+potentials?\s+(?:were\s+)?(?:converted|reported|referenced)\s+(?:to|vs\.?)\s*RHE|"
                r"calibrated\s+to\s+RHE|"
                r"referenced\s+to\s+RHE|"
                r"vs\.?\s*RHE(?:\s+scale)?|"
                r"(?:E\s*\(?\s*RHE\s*\)?|E\s*[_-]?\s*RHE|ERHE)\s*=\s*[^.\n]{0,120}"
                r"(?:pH|Ag/AgCl|Hg/HgO|SCE|NHE|SHE)"
                r"[^.\n]{0,120}"
                r")\b",
                re.IGNORECASE
            ),
        }


# =========================
# 2) organic_synthesis
# =========================
class OrganicSynthesisLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        temp_pat = r"(?:(?:-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|K)\b|\b(?:rt|r\.t\.|room\s+temperature|ambient\s+temperature)\b)"
        time_pat = r"\b\d+(?:\.\d+)?\s*(?:s|sec(?:onds?)?|min(?:utes?)?|h|hr(?:s)?|hours?|d|day(?:s)?)\b"

        metals = (
            r"(?:Pd|Ru|Rh|Ir|Ni|Co|Fe|Cu|Ag|Au|Pt|Zn|Al|Ti|Sn|Mn|Cr|Mo|W|V|"
            r"palladium|ruthenium|rhodium|iridium|nickel|cobalt|iron|copper)"
        )

        catalyst_metal_pat = rf"\b({metals}(?:\s*\([^)]+\))?|\bPd\(\s*OAc\s*\)\s*2\b|\bPd2\s*\(\s*dba\s*\)\s*3\b|\bRuCl3\b)\b"

        catalyst_loading_pat = catalyst_loading_pat = (
            r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>"
            r"mol\s*%|mol%|mol\s*percent|mol\s*percent\.?|"
            r"wt\s*%|wt%|weight\s*%|weight\s*percent)"
        )

        ligands_additives = (
            r"(?:PPh3|triphenylphosphine|BINAP|XPhos|SPhos|tBuXPhos|JohnPhos|Xantphos|dppf|dtbpf|"
            r"TMEDA|DMAP|NBS|NCS|TBAB|TBAI|LiCl|KF|CsF)"
        )

        bases = (
            r"(?:K2CO3|Cs2CO3|Na2CO3|K3PO4|Na3PO4|KOtBu|tBuOK|NaOtBu|Et3N|TEA|DIPEA|DBU|NaH|LiHMDS)"
        )

        solvents = (
            r"(?:THF|tetrahydrofuran|DCM|CH2Cl2|dichloromethane|toluene|benzene|MeCN|acetonitrile|DMF|DMSO|"
            r"1,4-dioxane|dioxane|EtOH|ethanol|MeOH|methanol|IPA|isopropanol|water|H2O|ethyl\s+acetate|EtOAc|"
            r"hexane|heptane|MTBE|diethyl\s+ether|ether|NMP|DMA|DMAC)"
        )

        reaction_types = (
            r"(?:Suzuki(?:\s+coupling)?|Miyaura|Heck|Sonogashira|Buchwald[-\s]?Hartwig|Negishi|Stille|Kumada|"
            r"Ullmann|hydrogenation|oxidation|reduction|amination|borylation|carbonylation|cyclization|"
            r"polymerization|esterification|amidation)"
        )

        workup = (
            r"(?:extraction|extracted|washed|filtration|filtered|dry(?:ing|ed)\s+over|concentrated\s+in\s+vacuo|"
            r"evaporated|recrystallization|recrystallized|column\s+chromatography|flash\s+chromatography|"
            r"purified\s+by|workup|quenched\s+with|partitioned)"
        )

        atmosphere = (
            r"\b(?:under|in)\s+(?:an?\s+)?(?:N2|nitrogen|Ar|argon|O2|oxygen|CO2|air|inert)\b"
            r"|\b(?:N2|nitrogen|Ar|argon|inert)\s+atmosphere\b"
            r"|\b(?:degassed|purged)\s+with\s+(?:N2|nitrogen|Ar|argon)\b"
        )

        self.rules = {
            "catalyst_metal": re.compile(catalyst_metal_pat, re.IGNORECASE),
            "catalyst_loading": re.compile(catalyst_loading_pat, re.IGNORECASE),
            "ligand_or_additive": re.compile(rf"\b{ligands_additives}\b", re.IGNORECASE),
            "base": re.compile(rf"\b{bases}\b", re.IGNORECASE),
            "solvent": re.compile(rf"\b{solvents}\b", re.IGNORECASE),
            "temperature": re.compile(temp_pat, re.IGNORECASE),
            "time": re.compile(time_pat, re.IGNORECASE),
            "atmosphere": re.compile(atmosphere, re.IGNORECASE),
            "reaction_type": re.compile(rf"\b{reaction_types}\b", re.IGNORECASE),
            "workup_method": re.compile(rf"\b{workup}\b", re.IGNORECASE),
        }


# =========================
# 3) cell_culture
# =========================
class CellCultureLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        temp_pat = r"(?:(?:-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃)\b|\b(?:room\s+temperature|ambient\s+temperature)\b)"
        time_pat = r"\b\d+(?:\.\d+)?\s*(?:min(?:utes?)?|h|hr(?:s)?|hours?|d|day(?:s)?)\b"

        media = (
            r"(?:DMEM/F12|DMEM|RPMI[-\s]?1640|MEM|IMDM|Ham['’]s\s*F[-\s]?12|F[-\s]?12|McCoy['’]s\s*5A|"
            r"Leibovitz['’]s\s*L[-\s]?15|Opti[-\s]?MEM|Neurobasal|BGJb|William['’]s\s*E)"
        )

        antibiotics = (
            r"(?:Pen/Strep|penicillin[-\s]*streptomycin|gentamicin|amphotericin\s*B|kanamycin|puromycin|"
            r"hygromycin|blasticidin|zeocin)"
        )

        common_cell_lines = (
            r"(?:HeLa|HEK[-\s]?293T?|RAW\s*264\.7|MCF[-\s]?7|A549|HUVEC|NIH\s*3T3|C2C12|Jurkat|U937|"
            r"THP[-\s]?1|PC12|CHO|Vero|MDCK|K562)"
        )
        cell_line_pat = (
            rf"\b{common_cell_lines}\b"
            r"|\bcell\s+line\s*[:=]?\s*[A-Za-z0-9][A-Za-z0-9\-\._]{1,25}\b"
            r"|\b[A-Za-z0-9][A-Za-z0-9\-\._]{1,25}\s+cells\b"
        )

        fbs_pat = r"\b\d+(?:\.\d+)?\s*%\s*(?:FBS|fetal\s+bovine\s+serum|serum)\b"

        seeding_density = (
            r"(?:\b\d+(?:\.\d+)?\s*(?:×|x|\*)\s*10\s*\^?\s*[-+]?\d+\s*(?:cells?|cell)\s*(?:/|per)\s*"
            r"(?:mL|ml|well|cm2|cm\^2|cm\s*2|dish|plate)\b)"
            r"|(?:\b\d+(?:\.\d+)?e[-+]?\d+\s*(?:cells?|cell)\s*(?:/|per)\s*(?:mL|ml|well|cm2|cm\^2|dish|plate)\b)"
        )

        co2_pat = r"\b\d+(?:\.\d+)?\s*%\s*CO\s*2\b"

        culture_duration = (
            rf"\b(?:incubat(?:ed|ion)|cultured|maintained)\b[^.\n]{{0,60}}{time_pat}"
            rf"|{time_pat}\s*(?:of\s+)?(?:incubation|culture)\b"
        )

        passage_pat = r"\bP\s*[-]?\s*\d+\b|\bpassage\s*(?:number)?\s*[:=]?\s*\d+\b"

        culture_format = (
            r"\b(?:2D|3D|monolayer|adherent|suspension|spheroid(?:s)?|organoid(?:s)?|scaffold(?:s)?|"
            r"hydrogel|Matrigel|transwell)\b"
        )

        self.rules = {
            "medium": re.compile(rf"\b{media}\b", re.IGNORECASE),
            "FBS_concentration": re.compile(fbs_pat, re.IGNORECASE),
            "cell_line": re.compile(cell_line_pat, re.IGNORECASE),
            "antibiotics": re.compile(rf"\b{antibiotics}\b", re.IGNORECASE),
            "seeding_density": re.compile(seeding_density, re.IGNORECASE),
            "incubation_temperature": re.compile(
                rf"\b(?:incubat(?:ed|ion)|cultured)\b[^.\n]{{0,50}}{temp_pat}|\b37\s*(?:°\s*C|℃)\b",
                re.IGNORECASE
            ),
            "CO2_percent": re.compile(co2_pat, re.IGNORECASE),
            "culture_duration": re.compile(culture_duration, re.IGNORECASE),
            "passage_number": re.compile(passage_pat, re.IGNORECASE),
            "culture_format": re.compile(culture_format, re.IGNORECASE),
        }


# =========================
# 4) immunotechnology (偏 WB/免疫印迹语境)
# =========================
class ImmunotechnologyLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        gel_percent = r"\b\d+(?:\.\d+)?\s*%\s*(?:SDS[-\s]*PAGE|gel)\b"
        loading_amt = r"\b\d+(?:\.\d+)?\s*(?:μg|ug|mg)\b"

        sample_type = (
            r"\b(?:tissue\s+lysate|cell\s+lysate|whole\s+cell\s+lysate|serum|plasma|supernatant|"
            r"homogenate|nuclear\s+extract|cytosolic\s+fraction)\b"
        )

        loading_control = (
            r"\b(?:GAPDH|β[-\s]?actin|beta[-\s]?actin|ACTB|α[-\s]?tubulin|tubulin|HPRT|Histone\s+H3)\b"
        )

        membrane = r"\b(?:PVDF|nitrocellulose|NC\s+membrane|PVDF\s+membrane|nitrocellulose\s+membrane)\b"

        blocking = (
            r"\b(?:\d+(?:\.\d+)?\s*%\s*(?:BSA|non[-\s]?fat\s+milk|skim\s+milk)|blocking\s+buffer|casein)\b"
        )

        primary_dil = r"\bprimary\s+antibody\b[^.\n]{0,80}\b\d+\s*:\s*\d+\b"
        secondary_dil = r"\bsecondary\s+antibody\b[^.\n]{0,80}\b\d+\s*:\s*\d+\b"

        replicates = r"\b(?:n\s*=\s*\d+|triplicate|duplicate|in\s+triplicate|biological\s+replicates?|technical\s+replicates?)\b"

        quant = r"\b(?:ImageJ|FIJI|Fiji|densitometr(?:y|ic)|Image\s*Lab|Quantity\s+One|Odyssey|GraphPad)\b"

        self.rules = {
            "sample_type": re.compile(sample_type, re.IGNORECASE),
            "loading_control": re.compile(loading_control, re.IGNORECASE),
            "membrane_type": re.compile(membrane, re.IGNORECASE),
            "blocking_buffer": re.compile(blocking, re.IGNORECASE),
            "primary_antibody_dilution": re.compile(primary_dil, re.IGNORECASE),
            "secondary_antibody_dilution": re.compile(secondary_dil, re.IGNORECASE),
            "gel_percentage": re.compile(gel_percent, re.IGNORECASE),
            "loading_amount": re.compile(
                rf"(?:loaded|loading|per\s+lane)[^.\n]{{0,40}}{loading_amt}|{loading_amt}\s*(?:protein)?\b",
                re.IGNORECASE
            ),
            "replicates_n": re.compile(replicates, re.IGNORECASE),
            "quantification_method": re.compile(quant, re.IGNORECASE),
        }


# =========================
# 5) inorganic_materials
# =========================
class InorganicMaterialsLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        temp = r"(?:-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|K)\b"
        time_pat = r"\b\d+(?:\.\d+)?\s*(?:min(?:utes?)?|h|hr(?:s)?|hours?|d|day(?:s)?)\b"

        target_phase = r"\b(?:target\s+phase|desired\s+phase|single\s+phase|phase\s+pure)\b[^.\n]{0,60}\b[A-Z][A-Za-z0-9\.\(\)\+\-]{1,40}\b"
        starting_materials = r"\b(?:starting\s+materials?|precursors?|raw\s+materials?|starting\s+powders?)\b[^.\n]{0,200}"

        pressing = r"\b(?:pressed|pressing|uniaxial|isostatic)\b[^.\n]{0,80}\b\d+(?:\.\d+)?\s*(?:MPa|GPa)\b"
        heating_rate = r"\b(?:heating\s+rate|ramp(?:ed)?\s+at)\b[^.\n]{0,40}\b\d+(?:\.\d+)?\s*(?:°\s*C|℃|K)\s*/\s*(?:min|h)\b|\b\d+(?:\.\d+)?\s*(?:°\s*C|℃|K)\s*/\s*(?:min|h)\b"

        calcination_temp = rf"\b(?:calcined|calcination|pre[-\s]?calcined|preheated|annealed)\b[^.\n]{{0,80}}{temp}"
        dwelling_time = rf"\b(?:dwell(?:ing)?\s*time|dwelling|held|maintained|soaked)\b[^.\n]{{0,80}}{time_pat}"

        atmosphere = (
            r"\b(?:under|in)\s+(?:flowing\s+)?(?:Ar|argon|N2|nitrogen|O2|oxygen|air|H2|hydrogen|CO2|vacuum|"
            r"forming\s+gas|reducing|inert)\b"
            r"|\b(?:Ar|argon|N2|nitrogen|O2|oxygen|air)\s+atmosphere\b"
        )

        mixing_method = (
            r"\b(?:mixed|mixing|homogenized|blended|ground|grinding|milled|milling|ball-?milled|ball-?milling)\b"
            r"[^.\n]{0,80}\b(?:powder(?:s)?|precursor(?:s)?|starting\s+material(?:s)?|mixture|pellet)\b"
            r"|"
            r"\b(?:mortar(?:\s+and\s+pestle)?|agate\s+mortar)\b[^.\n]{0,80}\b(?:ground|grinding)\b"
        )


        sintering_temp = rf"\b(?:sintered|sintering)\b[^.\n]{{0,80}}{temp}"
        sintering_time = rf"\b(?:sintered|sintering)\b[^.\n]{{0,120}}{time_pat}"

        self.rules = {
            "target_phase": re.compile(target_phase, re.IGNORECASE),
            "starting_materials": re.compile(starting_materials, re.IGNORECASE),
            "pressing_pressure": re.compile(pressing, re.IGNORECASE),
            "heating_rate": re.compile(heating_rate, re.IGNORECASE),
            "calcination_temp": re.compile(calcination_temp, re.IGNORECASE),
            "dwelling_time": re.compile(dwelling_time, re.IGNORECASE),
            "atmosphere": re.compile(atmosphere, re.IGNORECASE),
            "mixing_method": re.compile(mixing_method, re.IGNORECASE),
            "sintering_temp": re.compile(sintering_temp, re.IGNORECASE),
            "sintering_time": re.compile(sintering_time, re.IGNORECASE),
        }


# =========================
# 6) polymer_materials
# =========================
class PolymerMaterialsLocator(BaseLocator):
    def __init__(self):
        super().__init__()

        time_pat = r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec(?:onds?)?|min(?:utes?)?|h|hr(?:s)?|hours?)\b"

        process_goal = r"\b(?:cur(?:e|ing)|photocur(?:e|ing)|crosslink(?:ed|ing)|polymeriz(?:e|ed|ation|ing)|gelation)\b"

        resin_family = (
            r"\b(?:epoxy|acrylate|methacrylate|urethane|polyurethane|thiol[-\s]?ene|silicone|PDMS|vinyl\s+ester|"
            r"polyester|polyimide|PEGDA|PEG[-\s]?diacrylate|hydrogel)\b"
        )

        photoinitiator = (
            r"\b(?:TPO[-\s]?L?|BAPO|DMPA|LAP|camphorquinone|Irgacure\s*2959|Irgacure\s*819|Irgacure\s*1173|"
            r"Darocur\s*1173|Omnirad\s*819)\b"
        )

        photoinitiator_loading = (
            r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>"
            r"wt\s*%|wt%|w/w\s*%|w/v\s*%|phr|"
            r"mg\s*/\s*mL|g\s*/\s*L|mg\s*mL[-–]1|g\s*L[-–]1"
            r")\b"
        )

        wavelength = r"\b(?:wavelength|irradiat(?:ed|ion)|UV|light)\b[^.\n]{0,60}\b\d+(?:\.\d+)?\s*nm\b|\b\d+(?:\.\d+)?\s*nm\b"

        intensity = r"\b\d+(?:\.\d+)?\s*(?:mW|W)\s*/\s*cm\s*\^?\s*2\b"

        exposure = rf"\b(?:exposure\s+time|irradiat(?:ion|ed)|cured\s+for|UV\s+exposure)\b[^.\n]{{0,60}}{time_pat}|{time_pat}\s*(?:UV\s+)?(?:exposure|irradiation|curing)\b"

        atmosphere = r"\b(?:air[-\s]?capped|in\s+air|under\s+air|under\s+(?:N2|nitrogen|Ar|argon)|nitrogen\s+purged|inert\s+atmosphere|vacuum)\b"

        thickness = r"\b(?:film\s+thickness|thickness)\b[^.\n]{0,40}\b\d+(?:\.\d+)?\s*(?:nm|μm|um|mm)\b|\b\d+(?:\.\d+)?\s*(?:nm|μm|um|mm)\b"

        light_source = r"\b(?:LED|light[-\s]?emitting\s+diode|mercury\s+lamp|Hg\s+lamp|UV\s+lamp|xenon\s+lamp|Xe\s+lamp|laser)\b"

        self.rules = {
            "process_goal": re.compile(process_goal, re.IGNORECASE),
            "resin_family": re.compile(resin_family, re.IGNORECASE),
            "photoinitiator_type": re.compile(photoinitiator, re.IGNORECASE),
            "photoinitiator_loading": re.compile(photoinitiator_loading, re.IGNORECASE),
            "wavelength": re.compile(wavelength, re.IGNORECASE),
            "intensity": re.compile(intensity, re.IGNORECASE),
            "exposure_time": re.compile(exposure, re.IGNORECASE),
            "atmosphere": re.compile(atmosphere, re.IGNORECASE),
            "film_thickness": re.compile(thickness, re.IGNORECASE),
            "light_source_type": re.compile(light_source, re.IGNORECASE),
        }


# =========================
# dispatcher
# =========================
DOMAIN_LOCATORS = {
    "electrochemistry": ElectrochemistryLocator,
    "organic_synthesis": OrganicSynthesisLocator,
    "cell_culture": CellCultureLocator,
    "immunotechnology": ImmunotechnologyLocator,
    "inorganic_materials": InorganicMaterialsLocator,
    "polymer_materials": PolymerMaterialsLocator,
}


def process_and_split(domain_name: str, limit: int = 10000):
    base_dir = Path(f"seed_data/{domain_name}")
    input_path = base_dir / "paragraphs.json"

    fields_dir = base_dir / "re_extraction/fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    locator = DOMAIN_LOCATORS[domain_name]()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # data = data[:limit]

    field_data_map = {field: [] for field in locator.rules.keys()}
    all_results = []
    field_counts = Counter()

    for item in data:
        text = item.get("paragraph", "")
        evidence = locator.locate_all(text)  # 现在返回 {field: [match1, match2, ...]}

        if evidence:
            for field, matches in evidence.items():  # matches 是列表
                field_counts[field] += len(matches)  # 统计每个字段的总匹配数
                
                for match_info in matches:  # 遍历所有匹配
                    field_entry = {
                        "id": item.get("id", ""),
                        "doi": item.get("doi", ""),
                        "span": match_info["span"],  # 现在 match_info 是字典
                        "context": match_info["context"],
                    }
                    field_data_map[field].append(field_entry)

        item["evidence"] = evidence
        all_results.append(item)

    print(f"\n[INFO] {domain_name} 数据集处理完成，共 {len(all_results)} 条数据。")
    
    # 1) per-field files
    for field, records in field_data_map.items():
        field_file = fields_dir / f"{field}.json"
        with open(field_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        print(f"已生成字段文件: {field}.json (共 {len(records)} 条匹配)")

    # 2) overall file
    with open(base_dir / "re_extraction/evidence_span.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 3) stats
    total_matches = sum(field_counts.values())
    stats = {
        "domain": domain_name,
        "total_paragraphs": len(data),
        "paragraphs_with_evidence": sum(1 for x in all_results if x.get("evidence")),
        "total_matches": total_matches,
        "field_counts": dict(field_counts),
    }
    
    with open(base_dir / "re_extraction/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print("\n[DONE] 统计结果已存至 stats.json，字段明细已存至 fields/ 文件夹。")
    print(f"总计找到 {total_matches} 个匹配，分布在 {len(field_counts)} 个字段中。")


if __name__ == "__main__":
    process_and_split("electrochemistry")
    process_and_split("organic_synthesis")
    process_and_split("cell_culture")
    process_and_split("immunotechnology")
    process_and_split("inorganic_materials")
    process_and_split("polymer_materials")


# cd /home/liangwenliang/检索数据集/
# python scripts/structured_extraction/re_evidence_span.py



