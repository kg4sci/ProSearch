import json
from copy import deepcopy   


extract_template = """You are a professional information extractor specializing in {domain_name}.
|split|
Your task is to determine whether the paragraph contains at least one valid experimental record for this domain and, if it does, to extract one or more setup-level structured records.

A paragraph is valid only if it provides at least one piece of experimental information that can be mapped to the defined schema fields. This information may be stated explicitly or be recoverable through short, evidence-grounded domain reasoning. A paragraph is invalid if it mainly focuses on {irrelevant_def} and does not provide any extractable value for the defined fields.

If the paragraph is invalid, return:
{{"is_valid": false, "invalid_reason": one label from {invalid_labels}, "setups": []}}

If the paragraph is valid, return:
{{"is_valid": true, "invalid_reason": null, "setups": [...]}}

Use SETUP as the extraction unit. Create a new setup only when the paragraph explicitly describes a different system, composition, condition set, treatment, reaction setting, measurement configuration, material-role assignment, or other clearly distinct experimental configuration. Do not split one setup into multiple records just because the paragraph repeats it, comments on it, or reports outcomes for it.

Extract only the fields defined in the allowed field structure below. Do not extract other experimental details. If a mention does not clearly belong to one of these fields, omit it rather than forcing it into the nearest field. The goal is high-precision extraction of a small set of structured field values, not broad extraction of all information in the paragraph. When a field is decomposed into subfields, assign each piece of information only to its own subfield and do not repeat the same content across sibling subfields. In general, the "value" should contain only the core content for that field, while "evidence" should be an exact supporting span copied from the paragraph.

Use evidence-grounded scientific reasoning when necessary to map the paragraph to the defined fields. This includes standard domain interpretations, conventional experimental terminology, and short inference steps that are strongly supported by the paragraph. However, do not invent missing information, do not use multi-step speculation, and do not replace weak hints with confident conclusions. When the inference is weak or uncertain, omit the field.

You may lightly clean the "value" when the change is obvious and low-risk, such as normalizing spaces, symbols, hyphenation, or unit formatting. However, do not change the meaning of the extracted value.

Only extract information that is concrete enough to be useful for the defined fields. If a mention is vague, generic, weakly specified, not informative as a structured field value, or not clearly matched to one of the defined fields, omit it.

The schema shown below is only a reference to the available fields and their nesting structure. It is not a template that must be filled. In the actual output, include only fields that have extracted values supported by the paragraph. If a field or subfield has no value, omit it entirely. Never output null for missing fields. Never output empty objects, empty lists, or placeholder list items.

Always include "setup_id" for each extracted setup. Use setup IDs as "setup_1", "setup_2", and so on, following order of appearance.

Return JSON only, with no extra text.

Allowed field structure:
{output_schema}

Field guide:
{field_glossary}

Paragraph:
{paragraph}
"""

revise_template = """You are a professional information extractor specializing in {domain_name}.
|split|
You are given a paragraph, a field guide, an allowed field structure, and an initial extraction result.

Your task is to revise the initial extraction so that it is fully supported by the paragraph and consistent with the field definitions.

Use the paragraph as the source of truth. Do not assume the initial extraction is correct. Keep correct fields unchanged whenever possible, and make only the minimum necessary revisions. Remove unsupported or low-information values, correct wrongly assigned fields, shorten overlong values to the core field value, and add clearly supported missing fields when necessary.

Follow the field definitions carefully. If a value does not match the intended meaning of the field, remove it or move it only when the correct field assignment is explicit. If a mention is vague, generic, or not useful as a structured value, omit it. Do not infer missing information.

Keep the schema exactly consistent with the allowed field structure. Do not introduce new keys, new subfields, or new formats. In particular, do not split a field into extra parts such as separate value and unit fields unless that structure already exists in the allowed field structure. Keep each extracted field in the same value/evidence format.

Use sparse JSON. Omit unsupported fields entirely. Do not output null, empty objects, empty lists, or placeholder list items.

Preserve setup structure when it is already correct. Only merge or split setups when the paragraph clearly shows that the current setup boundaries are wrong.

Return only a JSON object in the form {{"setups": [...]}}, where "setups" is always a list, even when there is only one setup.

Field guide:
{field_glossary}

Allowed field structure:
{output_schema}

Paragraph:
{paragraph}

Initial extraction:
{stage1_extraction}
"""

DOMAIN_SPECS = {
    "electrochemistry": {
        "domain_name": "electrochemistry",
         "valid_def": (
            "at least one piece of electrochemistry-related experimental information that can be mapped to the defined schema fields, "
            "such as electrolyte composition, reference electrode, working electrode, target species, temperature, "
            "measurement method, scan rate, or applied potential, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "material synthesis, precursor preparation, thermal treatment, electrode fabrication or preparation, battery assembly, "
            "characterization-only content, or background/discussion content that does not provide any extractable value for the defined electrochemical schema fields"
        ),
        "invalid_labels": [
            "synthesis_or_preparation",
            "electrode_fabrication",
            "battery_related",
            "characterization",
            "background_or_discussion",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "electrolytes": [
                {
                    "chemical": {"value": None, "evidence": None},
                    "concentration": {"value": None, "evidence": None}
                }
            ],
            "reference_electrode": {"value": None, "evidence": None},
            "working_electrode": {"value": None, "evidence": None},
            "target_species": {"value": None, "evidence": None},
            "temperature": {"value": None, "evidence": None},
            "measurement_method": {"value": None, "evidence": None},
            "scan_rate": {"value": None, "evidence": None},
            "applied_potential": {"value": None, "evidence": None}
        } ,
       "field_glossary": "\n".join([
            "electrolytes: electrolyte components explicitly used in the electrochemical system. Use one entry per chemically distinct component. The chemical subfield should contain only the component identity, such as EMImBF4, EMImCl, CH3CN, or water, rather than a combined phrase containing multiple components. The concentration subfield should contain only the explicitly stated amount or fraction for that component when given. Solvents, added water, and additives should be represented as separate electrolyte entries when they are part of the electrolyte composition. Do not merge multiple distinct concentration conditions into a single setup when they represent separate experimental compositions or a parameter sweep.",
            "reference_electrode: the reference electrode explicitly used in the setup, such as Ag/AgCl, Ag|AgCl pseudoreference, Ag in 0.1 M AgNO3 in CH3CN, or hydrogen-saturated palladium wire. Extract the reference electrode as a single usable identity phrase supported by the local evidence. Do not invent missing details. Normalization of reference-electrode family can be handled later.",
            "working_electrode: the explicitly stated working electrode identity used in the setup, such as platinum wire, Pt wire, Bi(111) crystal, or interdigitated platinum electrodes. Use this field only when the working electrode identity itself is explicitly given. Do not fill this field with placeholders or generic apparatus descriptions such as three-electrode cell when the working electrode is not specified.",
            "target_species: the explicitly stated analyte, reactant, or key electroactive species being studied, detected, reduced, oxidized, or otherwise involved as the main target of the electrochemical measurement, such as O2, CO2, H2O2, glucose, nitrate, or a named redox-active species. Do not place ordinary supporting electrolyte components here.",
            "temperature: the explicitly stated experimental temperature, such as 25 °C or room temperature. Use this field only for actual thermal conditions. Do not place dry/wet labels, water-content conditions, or electrolyte-conditioning descriptions here.",
            "measurement_method: the electrochemical measurement method explicitly used in the setup, such as cyclic voltammetry, electrochemical impedance spectroscopy, chronoamperometry, or conductance measurements. Multiple clearly stated methods may be included together when they belong to the same setup.",
            "scan_rate: the explicitly stated scan rate used in cyclic voltammetry, such as 100 mV/s. Use this field only when a scan rate is clearly stated.",
            "applied_potential: the explicitly stated applied potential, fixed potential, or electrochemical potential window used in the measurement, such as -0.8 V, 0.2 to 1.0 V, or 1.2 V vs Ag/AgCl. Extract the potential as stated, including reference information when given. Do not infer or convert potentials during extraction."
        ])
    },

    "organic_synthesis": {
        "domain_name": "organic synthesis",
       "valid_def": (
            "at least one piece of reaction-related experimental information that can be mapped to the defined schema fields, "
            "such as transformation type, substrate, catalyst, catalyst loading, ligand, base, additive, solvent, "
            "temperature, time, or activation mode, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "background discussion, broad mechanistic interpretation, product characterization-only content, review-style summary, "
            "or other content that does not provide any extractable value for the defined reaction schema fields"
        ),
        "invalid_labels": [
            "background",
            "mechanistic_discussion",
            "characterization",
            "review_or_summary",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "transformation_type": {"value": None, "evidence": None},
            "substrates": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "catalysts": [
                {
                    "name": {"value": None, "evidence": None},
                    "loading": {"value": None, "evidence": None}
                }
            ],
            "ligands": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "bases": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "additives": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "solvents": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "activation_mode": {"value": None, "evidence": None}
        },
        "field_glossary": "\n".join([
            "transformation_type: the reaction class or transformation type carried out in this setup, such as Suzuki coupling, cross-coupling, hydrogenation, oxidation, cyclization, or amidation. Use this field when the transformation is explicitly stated or directly supported by short, evidence-grounded chemical reasoning from a clear and diagnostic reaction pattern. Do not assign a transformation type from weak or ambiguous cues alone.",

            "substrates: the key stoichiometric starting materials or reaction partners that are transformed into the product in this setup. Use one entry per substrate identity. Extract only the substrate identity itself, not the full amount phrase or surrounding procedural wording. Omit local compound labels or paper-specific identifiers such as 1a, 10b, or 14 when they are not part of the chemical identity itself. Do not place catalysts, ligands, bases, solvents, or routine workup reagents here.",

            "catalysts: catalytic species used in the reaction. The name subfield should contain only the catalyst identity, such as Pd(PPh3)4, PdCl2(PPh3)2, CuTC, or RuCl3. The loading subfield should contain only the explicitly stated catalyst loading, such as 5 mol %, 10 mol %, or 0.3 mmol, when given. Do not create a separate metal subfield.",

            "ligands: ligand identities used in the reaction, such as PPh3, BINAP, bpy, or XPhos. Extract ligands when they are explicitly added as ligands or when the ligand identity is directly recoverable from a named catalyst complex. Do not include the full catalyst description here; extract only the ligand identity itself.",

            "bases: bases used in the reaction, such as Cs2CO3, Na2CO3, K3PO4, or Et3N. Extract only the base identity itself, not the full amount phrase.",

            "additives: non-base, non-solvent, non-catalyst additives or promoters that play a meaningful reaction role, such as oxidants, reductants, acids, salts, or other explicitly added functional additives. Extract only the additive identity itself. Do not place routine solvents, bases, catalysts, or workup reagents here.",

            "solvents: solvents or solvent components used as the reaction medium, such as DME, toluene, THF, water, or H2O. Use one entry per solvent component. Extract only the solvent identity itself, not the volume or the full phrase. It is acceptable to normalize common full-name/abbreviation pairs later during standardization.",
            
            "activation_mode: the high-level reaction-driving or activation mode used in this setup, such as thermal, photochemical, electrochemical, microwave, mechanochemical, or sonochemical. This field may be assigned when the mode is explicitly stated or directly supported by clear local evidence, such as reflux, heating at a specified temperature, light irradiation, electrolysis, microwave irradiation, ball milling, or ultrasound. Do not infer an activation mode from weak or ambiguous procedural cues alone."
        ])
    },

    "cell_culture": {
        "domain_name": "cell culture",
         "valid_def": (
            "at least one piece of cell-culture-related experimental information that can be mapped to the defined schema fields, "
            "such as cultured cell identity, cell source type, basal medium, serum identity or concentration, substrate or coating, "
            "seeding density, culture mode, differentiation state, or confluence, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "assay-only content, downstream analysis, imaging-only settings, ethics or sample sourcing statements, general instrumentation, "
            "or other content that does not provide any extractable value for the defined cell culture schema fields"
        ),
        "invalid_labels": [
            "assay",
            "downstream_analysis",
            "imaging",
            "ethics_or_sourcing",
            "general_methods",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "cultured_cells": [
                {
                    "cell_type": {"value": None, "evidence": None}
                }
            ],
            "cell_source_type": {"value": None, "evidence": None},
            "culture_medium": {
                "basal_medium": {"value": None, "evidence": None},
                "serum": {
                    "identity": {"value": None, "evidence": None},
                    "concentration": {"value": None, "evidence": None}
                }
            } ,
            "culture_substrate_or_coating": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "seeding_density": {"value": None, "evidence": None},
            "culture_mode": {"value": None, "evidence": None},
            "differentiation_state": {"value": None, "evidence": None},
            "confluence": {"value": None, "evidence": None}
        },
        "field_glossary": "\n".join([
            "cultured_cells: the cells actually being cultured in this setup. Use one entry per cell type. cell_type should contain the canonical minimal cell identity needed for stable distinction, in a compact usable form. For established named cell lines, prefer the standard line name itself, such as SH-SY5Y, rather than longer descriptive phrases. Omit generic shell words such as 'cells' or 'cell line' unless needed for distinction. Retain extra descriptors only when they are necessary to distinguish the cell identity.",

            "cell_source_type: the source category of the cultured cells. Use concise labels such as primary, cell line, immortalized, stem-cell-derived, or patient-derived. Infer a label only when it is explicitly stated or directly supported by short evidence-grounded inference. The evidence span should directly support the assigned label and should be as complete as possible, preferably including the key local phrase that justifies the category, rather than a short fragment that is only indirectly related.",

            "culture_medium: the culture medium explicitly used in the setup. basal_medium should contain only the main basal medium identity, such as DMEM, RPMI-1640, IMDM, DMEM/F12, or Ham's F12 and MEM (1:1), rather than the full medium phrase. Omit formulation details, supplements, and additives such as high glucose or L-glutamine unless they are part of the basal medium identity itself. serum should capture the serum used in the medium. identity should contain only the canonical serum identity, such as FBS, FCS, or horse serum. concentration should contain only the serum amount, such as 10% or 2%, without extra qualifiers when they do not change the core value.",

            "culture_substrate_or_coating: The material used for cell attachment, support, or embedding, such as 'Collagen I' or 'Matrigel'. Extract only the material identity. Omit concentrations, preparation details, and vessel descriptions.",

            "seeding_density: The initial seeding or plating amount used to start the culture. The value should include the stated cell number together with its denominator or culture unit when available, such as '2 x 10^4 cells/cm^2', '1 x 10^5 cells/mL', '5 x 10^4 cells/well', or '1 x 10^6 cells/flask'. If the text gives only an isolated number that is not clearly the initial seeding or plating amount, omit this field.",

            "culture_mode: the organizational culture format of the setup, such as 2D, 3D, spheroid, organoid, suspension, or co-culture, only when directly stated or directly supported by short local evidence. Do not infer culture mode from vessel type or vessel size alone. Flask, dish, plate, well format, T25, T75, or similar vessel descriptions are not sufficient evidence for culture_mode.",

            "differentiation_state: Use 'differentiated' or 'undifferentiated' only when the text explicitly states the state or clearly ties the setup to a differentiation protocol and resulting state. Do not assume that every treatment or induction implies differentiation.",

            "confluence: extract only the core confluence value, range, or state, such as 80%, 80-85%, or near confluence. Omit approximation markers and surrounding procedural wording when they do not change the core value."
        ])
    },

    "immunotechnology": {
        "domain_name": "immunotechnology",
        "valid_def": (
            "at least one piece of Western blot or immunoblot-related experimental information that can be mapped to the defined schema fields, "
            "such as protein target, target form, normalization control, sample type, lysis buffer, blocking-solution identity or concentration, "
            "gel percentage, or loading amount, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "non-Western-blot assays such as immunofluorescence, immunohistochemistry, flow cytometry, or ELISA; general methods, "
            "figure description, or other content that does not provide any extractable value for the defined Western blot schema fields"
        ),
        "invalid_labels": [
            "non_wb_assay",
            "general_methods",
            "figure_only",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "protein_targets": [
                {
                    "target": {"value": None, "evidence": None},
                    "target_form": {"value": None, "evidence": None}
                }
            ],
            "normalization_control": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "sample_type": {"value": None, "evidence": None},
            "lysis_buffer": {"value": None, "evidence": None},
            "blocking_solution": {
                "identity": {"value": None, "evidence": None},
                "concentration": {"value": None, "evidence": None}
            },
            "gel_percentage": {"value": None, "evidence": None},
            "loading_amount": {"value": None, "evidence": None}
        },
        "field_glossary": "\n".join([
            "protein_targets: the protein targets intentionally probed in the Western blot. Use one entry per target protein. The target subfield should contain only the protein target identity, such as Akt, ERK, MARCO, TIM4, SAV1, or MOB1A, rather than the full antibody product description. The target_form subfield should describe the target form only when it is explicitly stated or directly supported by short evidence-grounded inference, using concise values such as phospho or total. For example, p-Akt or phosphorylated Akt may support phospho, and total Akt may support total. If the form is not clearly stated, leave target_form empty rather than guessing. Do not place normalization or loading-control proteins here when the paragraph clearly assigns them a control role.",

            "normalization_control: proteins explicitly used as loading controls or normalization controls in the blot, such as GAPDH, beta-actin, ACTB, tubulin, or vinculin. Extract only the control protein identity, not the surrounding sentence. Use this field when the paragraph clearly indicates a normalization or control role, such as normalized to GAPDH, served as internal control, or used as loading control.",

            "sample_type: the sample category used for blotting, such as cell lysate, whole cell lysate, tissue lysate, total protein extract, nuclear extract, cytoplasmic extract, serum, or plasma. Extract the minimal sample identity, not the full procedural phrase. Keep this field focused on the sample category itself and do not merge it with lysis-buffer identity, vendor information, or detailed preparation steps.",

            "lysis_buffer: the explicitly stated lysis or extraction buffer used to prepare the sample, such as RIPA buffer, RIPA lysis buffer, NP-40 lysis buffer, or urea buffer. Extract only the buffer identity in a compact usable form. Omit vendor names, catalogue information, and surrounding preparation procedure unless they are required to identify the buffer itself.",

            "blocking_solution: the blocking solution used before antibody incubation. When both blocking reagent identity and concentration are stated, separate them across the corresponding subfields. identity should contain only the blocking reagent or named blocking solution, such as BSA, nonfat dry milk, or Intercept TBS Blocking Buffer. concentration should contain only the stated amount, such as 5%. Do not place buffer context such as TBST or PBS into a separate structured subfield unless it is essential for identifying the blocking solution itself.",

            "gel_percentage: the explicitly stated SDS-PAGE gel percentage or percentage range, such as 8%, 10%, or 10-12%. Extract only the gel percentage information itself.",

            "loading_amount: the explicitly stated sample or protein loading amount per lane or well, such as 20 ug, 20 ug/well, 30 ug, or 50 ug. Extract only the loading amount itself, not the surrounding procedural wording."
        ])
    },

    "inorganic_materials": {
        "domain_name": "inorganic materials",
       "valid_def": (
            "at least one piece of solid-state inorganic-materials-related experimental information that can be mapped to the defined schema fields, "
            "such as target material, precursor, dopant or additive, synthesis route, atmosphere, heating rate, pressing pressure, "
            "sintering condition, or post-treatment condition, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "characterization-only content, property testing, non-solid-state or non-target synthesis content, thin-film deposition, "
            "or other content that does not provide any extractable value for the defined solid-state synthesis schema fields"
        ),
        "invalid_labels": [
            "characterization",
            "property_test",
            "thin_film",
            "non_target_synthesis",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "target_material": {"value": None, "evidence": None},
            "precursors": [
                {
                    "value": None,
                    "evidence": None
                }
            ],
            "dopant_or_additive": [
                {
                    "identity": {"value": None, "evidence": None},
                    "amount": {"value": None, "evidence": None}
                }
            ],
            "synthesis_route": {"value": None, "evidence": None},
            "atmosphere": {"value": None, "evidence": None},
            "heating_rate": {"value": None, "evidence": None},
            "pressing_pressure": {"value": None, "evidence": None},
            "sintering": {
                "temperature": {"value": None, "evidence": None},
                "time": {"value": None, "evidence": None}
            },
            "post_treatment": {
                "type": {"value": None, "evidence": None},
                "temperature": {"value": None, "evidence": None},
                "time": {"value": None, "evidence": None}
            }
        },
        "field_glossary": "\n".join([
            "target_material: the target inorganic material system prepared or studied in this setup. Extract the minimal material identity needed to distinguish the material system, such as Al2O3, ZTA, SiC, or AlN-Al2O3 composite, rather than the full descriptive sentence.",

            "precursors: the explicitly stated precursor materials or starting powders used to prepare the target material. Use one entry per precursor identity. Extract only the precursor identity itself, not the surrounding procedural wording.",

            "dopant_or_additive: dopants, additives, or sintering aids explicitly introduced into the material system when they function as composition-changing modifiers or meaningful processing additives. When both additive identity and amount are stated, separate them across the corresponding subfields. identity should contain only the additive identity, such as Y2O3, MgO, or CaF2. amount should contain only the stated amount, such as 5 wt% or 2 mol%. Do not output vague placeholders such as additives or unspecified additive. Do not use this field for generic solvents, liquid media, or routine processing aids such as deionized water unless they are clearly treated as meaningful experimental variables.",

            "synthesis_route: the high-level synthesis or consolidation route used to produce the material, such as spark plasma sintering, hot-pressure sintering, pressureless sintering, solid-state reaction, or sol-gel. Extract only the route identity itself. Do not include atmosphere, pressure, temperature, time, mold information, or other processing details in this field. Do not use overly generic values such as sintering unless no more specific route identity is stated and the route itself is still experimentally meaningful.",

            "atmosphere: the explicitly stated processing atmosphere relevant to synthesis, consolidation, sintering, or post-treatment, such as air, vacuum, nitrogen, argon, or inert nitrogen atmosphere.",

            "heating_rate: the explicitly stated heating rate used in thermal processing, such as 10 °C/min or 100 °C/min. Extract only the rate itself.",

            "pressing_pressure: the explicitly stated pressing or consolidation pressure used in the relevant processing step, such as 12 MPa, 20 MPa, or 45 MPa. Extract only the pressure value itself.",

            "sintering: the explicitly stated sintering conditions. temperature should contain only the sintering temperature, and time should contain only the sintering dwell or holding time. Do not merge multiple distinct temperatures or times into a single value when they represent separate experimental conditions. Split them into separate setups when the paragraph describes distinct condition sets or a parameter sweep.",

            "post_treatment: distinct thermal-treatment steps performed before or after the main sintering step, such as annealing, oxidation, calcination, or heat treatment, when they function as meaningful processing conditions. type should contain the treatment identity itself. temperature should contain the treatment temperature when stated. time should contain the treatment duration when stated. Exclude routine drying, storage, or aging unless they are clearly treated as decision-relevant thermal treatments."
        ])
    },

    "polymer_materials": {
        "domain_name": "polymer materials",
        "valid_def": (
            "at least one piece of photocuring- or photopolymerization-related experimental information that can be mapped to the defined schema fields, "
            "such as polymer system, polymer concentration, photoinitiator, initiator loading, crosslinking agent, crosslinker loading, "
            "wavelength, light intensity, exposure time, or light source type, either explicitly or through short, evidence-grounded domain reasoning"
        ),
        "irrelevant_def": (
            "resin or monomer synthesis, characterization-only content, mechanical or rheology testing, "
            "or other content that does not provide any extractable value for the defined photocuring schema fields"
        ),
        "invalid_labels": [
            "synthesis",
            "characterization",
            "property_or_rheology_test",
            "no_extractable_fields",
            "other"
        ],
        "setup_schema": {
            "setup_id": "setup_1",
            "polymer_system": {"value": None, "evidence": None},
            "polymer_concentration": {"value": None, "evidence": None},
            "photoinitiators": [
                {
                    "name": {"value": None, "evidence": None},
                    "loading": {"value": None, "evidence": None}
                }
            ],
            "crosslinking_agent": [
                {
                    "name": {"value": None, "evidence": None},
                    "loading": {"value": None, "evidence": None}
                }
            ],
            "wavelength": {"value": None, "evidence": None},
            "intensity": {"value": None, "evidence": None},
            "exposure_time": {"value": None, "evidence": None},
            "light_source_type": {"value": None, "evidence": None}
        },
        "field_glossary": "\n".join([
            "polymer_system: the main photocurable polymer, resin, or network system being cured, such as GelMA, PEGDA, or GelMA-NBNAGA. Extract only the main system identity itself, not the concentration, photoinitiator, or full formulation phrase.",

            "polymer_concentration: the explicitly stated concentration of the main polymer system, such as 10 wt%, 15% w/v, or 20 wt%. Extract only the concentration value itself, including its unit. Do not infer or convert units during extraction.",

            "photoinitiators: photoinitiator entries used in the curing setup. The name subfield should contain only the initiator identity, such as LAP, Irgacure 2959, or eosin Y. The loading subfield should contain only the explicitly stated loading value, including its original unit, without repeating the full descriptive phrase in both places. Do not infer or convert units during extraction.",

            "crosslinking_agent: explicitly added crosslinking components or other distinct reactive formulation additives used together with the main polymer system, such as PEGDA, TEA, NVP, or another clearly named auxiliary reactive component, when they are described as separate formulation ingredients. The name subfield should contain only the component identity, and the loading subfield should contain only the explicitly stated loading value when given. Do not duplicate the main polymer system or the photoinitiator here.",

            "wavelength: the explicitly stated irradiation wavelength in a concrete usable form, such as 365 nm, 405 nm, or 395-400 nm. Extract only explicitly stated wavelength values or narrow wavelength ranges. Do not fill this field with color descriptions or with phrases indicating that wavelength is not specified.",

            "intensity: the explicitly stated light intensity or irradiance in a concrete usable form. Preserve the original value and unit as written, such as mW/cm^2 or mW/mm^2. Do not convert units during extraction.",

            "exposure_time: the explicitly stated irradiation or exposure time in a concrete usable form.",

            "light_source_type: the explicitly stated light source type, such as LED, UV LED, mercury lamp, projector, laser, or another clearly named source type. Extract only the source identity itself, not the wavelength when wavelength is captured separately. Prefer the most specific source identity directly supported for the setup, and do not merge multiple different source descriptions into one value."
        ])
    }
}

BASE_TOP_LEVEL_SCHEMA = {
    "is_valid": True,
    "invalid_reason": None,
    "setups": []
}

def build_output_schema(domain_spec):
    schema = deepcopy(BASE_TOP_LEVEL_SCHEMA)
    schema["setups"] = [deepcopy(domain_spec["setup_schema"])]
    return schema

def render_extract_prompt(domain, paragraph):
    spec = DOMAIN_SPECS[domain]
    output_schema = build_output_schema(spec)
    promtp = extract_template.format(
        domain_name=spec["domain_name"],
        valid_def=spec["valid_def"],
        irrelevant_def=spec["irrelevant_def"],
        invalid_labels=spec["invalid_labels"],
        output_schema=json.dumps(output_schema, ensure_ascii=False, indent=2),
        field_glossary=spec["field_glossary"],
        paragraph=paragraph
    )
    system_prompt, user_prompt = promtp.split("|split|")
    return system_prompt, user_prompt

def render_revise_prompt(domain, paragraph, stage1_extraction):
    promtp = revise_template.format(
        paragraph=paragraph,
        domain_name=DOMAIN_SPECS[domain]["domain_name"],
        field_glossary=DOMAIN_SPECS[domain]["field_glossary"],
        output_schema=json.dumps(DOMAIN_SPECS[domain]["setup_schema"], ensure_ascii=False, indent=2),
        stage1_extraction=stage1_extraction
    )
    system_prompt, user_prompt = promtp.split("|split|")
    return system_prompt, user_prompt

if __name__ == "__main__":
    # print(render_extract_prompt("polymer_materials", "paragraph 1"))
    print(render_revise_prompt("polymer_materials", "paragraph 1",'stage_1_extraction'))
    
# python scripts/structured_extraction/llm_extract_prompt.py

