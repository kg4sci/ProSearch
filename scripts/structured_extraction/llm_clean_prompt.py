import json

PROMPT_TEMPLATE = """
You are a careful scientific data normalizer.
|split|
I will give you the domain, the field name, a field-specific requirement, and a batch of items. Each item contains an item_id, a current value list, and its evidence. Your task is to normalize the current values for this field so that obvious surface variation is reduced, while distinctions that may still matter are preserved.

Use the current value list as the anchor for each item. Read the evidence only to understand, clarify, or clean those same values in context. Do not use the evidence to introduce a new value that is not already present in the current value list, and do not replace the current item with another mention that happens to appear nearby in the sentence. If the evidence contains multiple related mentions, keep only what clearly corresponds to the current item itself.

Follow the field requirement closely. Keep only content that genuinely belongs to the current field, and remove surrounding explanation, supplier information, concentrations, procedural wording, result-oriented language, and other details that do not belong in the normalized value, unless the field requirement makes that detail necessary for preserving a meaningful distinction. When a current value already expresses a clean and field-appropriate concept, keep it with only minimal normalization. When different expressions clearly refer to the same concept, use one consistent canonical form. At the same time, do not merge values that may still differ in a way that matters.

If a current value explicitly contains more than one value for the same field, you may split it into separate normalized values. But do not create extra values just because the evidence mentions other related items. If a value is clearly not appropriate for the current field, return an empty list for that item. If it may belong to the field but cannot be normalized confidently, keep a minimally cleaned version rather than making a strong inference.

Return a single JSON object only. Each key must be an item_id from the input batch, and each value must be a JSON array of canonical strings. Use an empty array when nothing should be kept. Remove duplicates within each array and sort the final values alphabetically.

Domain: {domain}
Field name: {slot_name}
Field requirement: {slot_requirement}
Batch items:
{batch_items}

Return the JSON object only.
"""



FIELD_REQUIREMENTS = {
    "cell_culture": {
        "cultured_cells.cell_type": (
            "Normalize to one or more canonical cell-identity values needed for stable distinction in this field. "
            "Prefer the standard established cell name itself when it is sufficient as a stable identity, such as SH-SY5Y, HepG2, 16HBE14o-, or 1321N1. "
            "For concise biological cell types without a stable line name, keep the canonical biological identity itself, such as erythroblasts or primary human hepatocytes. "
            "Keep extra descriptors only when they are necessary to distinguish a biologically different cell identity, such as species, primary status, stem-cell class, or a specific differentiated form. "
            "For opaque sample IDs or line-like codes, keep the accompanying cell-class descriptor when the code alone would be insufficient or ambiguous as a cell identity, such as 19-9-11 hiPSC rather than 19-9-11 alone. "
            "Remove generic shell words such as cells, cell line, cell culture, and culture unless they are needed for distinction. "
            "Do not keep source information, treatment status, resistance status explained outside the canonical name, labeling status, passage information, vendor information, or procedural wording in this field. "
            "Do not keep pure group labels, condition labels, or experimental signs such as +ST or -ST unless the evidence clearly shows that they are stable named cell identities, clones, or sublines. "
            "If multiple explicit cell types are present, split them into separate canonical values. "
            "If two surface forms refer to the same cell identity, keep only the more specific established canonical form."
        ),

        "cell_source_type": (
            "Normalize to one or more coarse cell-source category labels. "
            "Allowed canonical values are primary, cell line, immortalized, stem-cell-derived, and patient-derived. "
            "Return a list when different cultured cell populations in the same mention belong to different source categories. "
            "Use the most specific supported category for each cell population: use immortalized instead of cell line when immortalization is explicit; "
            "use patient-derived instead of primary when patient origin is explicit; use stem-cell-derived when the cells are explicitly described as iPSC-derived, ESC-derived, or stem-cell-derived. "
            "Map donor-derived or healthy-donor cells to primary unless patient origin is explicitly stated. "
            "Map cells derived from a named cell line to cell line when this is clearly supported. "
            "Keep only the source category labels themselves, and remove cell names, vendor names, catalog information, storage status, treatment details, species labels, and procedural wording. "
            "Do not use co-culture, commercial, cryopreserved, culture collection, differentiated, human, glycerol stock, bacterial strain, or in vitro culture as source-category labels. "
            "If the text does not clearly support any allowed source category, return []."
        ),

        "culture_medium.basal_medium": (
            "Keep only the identity of the basal culture medium or media used in the setup. "
            "Retain standard basal media such as DMEM, RPMI-1640, IMDM, MEM, Ham's F12, L-15, MS, or other explicitly named medium identities when they function as the main culture medium. "
            "If the current value explicitly describes a mixture of multiple basal media, split and return each medium identity separately. "
            "Remove concentration, strength, percentage, supplier information, serum-free or additive-free wording, and other formulation details unless they are part of the fixed medium name itself. "
            "Do not keep serum, supplements, antibiotics, growth factors, oxygen conditions, conditioned media, generic references such as culture medium, or gelling agents such as agar unless they are part of a named medium identity."
        ),

        "culture_medium.serum.identity": (
            "Keep only the core serum type represented by the current value in the culture medium. "
            "Retain common serum types such as fetal bovine serum, fetal calf serum, bovine calf serum, "
            "bovine serum, horse serum, or human serum. "
            "Remove concentration, supplier information, heat-inactivation status, blood-group labels, stripped status, "
            "and other descriptive or formulation details that do not change the core serum type. "
            "If the value indicates the absence of serum or of a specific serum type, such as serum-free, without serum, without FBS, no serum, or FBS-free, return []. "
            "Do not keep proteins, serum substitutes, serum replacements, or other additives that are not themselves serum types."
        ),

        "culture_medium.serum.concentration": (
            "Keep only concentration expressions that are explicitly stated for serum, such as FBS, serum, human serum, horse serum, or serum-free conditions. "
            "Normalize concentrations to canonical forms such as 10%, 2%, or 0-10%. "
            "Map serum-free or absence of serum to 0%. "
            "If multiple explicit serum concentrations are present, return them as separate values. "
            "If the evidence provides both a broad range and explicit enumerated concentrations, prefer the explicit concentrations. "
            "Remove the serum name itself and discard medium descriptions, additives, non-serum supplements, replacement reagents, control labels, timing notes, and procedural wording. "
            "Do not keep concentrations for DMSO, ethanol, ITS, supplements, or other non-serum components. "
            "If the concentration is not clearly supported as a serum concentration, return []."
        ),

        "culture_substrate_or_coating": (
            "Keep only the specific material, matrix, surface, or named engineered substrate actually used for cell attachment, coating, support, or embedding, "
            "such as Collagen I, Matrigel, gelatin, laminin, fibronectin, polystyrene, tissue culture plastic, PPy, Nichoid, or a named scaffold formulation such as 0C10P or 0C10P+BG-MPs. "
            "Remove concentrations, dimensions, pore sizes, vessel or insert descriptions, vendor names, treatment actions, and procedural wording. "
            "Exclude generic cultureware or support terms that do not provide a specific substrate identity, such as plate, well, dish, Petri dish, insert, Transwell insert, scaffold, scaffolds, or film alone. "
            "Keep a named scaffold or engineered substrate only when its identity itself is explicit; otherwise return []. "
            "Normalize equivalent naming variants to one canonical form, but do not merge distinct materials. "
            "If multiple explicit materials are independently used as substrates or coatings, split them into separate canonical values; "
            "however, keep a named composite scaffold or substrate formulation as one canonical value when it functions as a single substrate identity."
        ),

        "seeding_density": (
            "The initial seeding or plating amount used to start the culture. "
            "Keep the stated cell number together with its denominator or culture unit when available, such as "
            "'2 x 10^4 cells/cm^2', '1 x 10^5 cells/mL', '5 x 10^4 cells/well', or '1 x 10^6 cells/flask'. "
            "Range values may also be kept when they clearly describe the initial seeding or plating amount. "
            "Counts without a denominator may be kept only when the text explicitly states that they were seeded, plated, or inoculated as the starting amount. "
            "Do not keep isolated numbers that are not clearly the initial seeding or plating amount. "
            "Do not extract later culture density, final concentration, maintained density, reached density, suspended concentration, or other culture-stage cell concentration values unless the text explicitly presents them as the initial seeding or plating amount."
        ),

        "culture_mode": (
            "Normalize to coarse retrieval-oriented culture descriptors. "
            "Allowed canonical values are 2D, 2.5D, 3D, spheroid, organoid, suspension, chip, Transwell, ALI, co-culture, and monoculture. "
            "Return a list of one or more canonical values when multiple independent culture descriptors are clearly present in the original value or evidence. "
            "Do not create combined labels such as 3D scaffold, 2D chip, or 3D ALI. "
            "Normalize scaffold-based, embedded, or in-gel culture to 3D. "
            "Normalize chip-based systems to chip. "
            "Normalize Transwell or insert systems to Transwell. "
            "Normalize air-liquid interface culture to ALI. "
            "Keep co-culture or monoculture only when explicitly stated. "
            "Allow simple domain-consistent inference for common vessel-based adherent culture, such as dish, plate, well, flask, T25, or T75, which may be normalized to 2D when no stronger culture descriptor is present. "
            "Remove brand names, vessel sizes, procedural wording, and other non-mode details."
        ),

        "differentiation_state": (
            "Keep only the differentiation status of the cultured cells, using only these canonical labels: "
            "differentiated, undifferentiated, dedifferentiated, and partially differentiated. "
            "Map semantically equivalent expressions to one of these labels when the meaning is clear "
            "(for example, intermediate differentiation state, less differentiated, or less-differentiated -> partially differentiated). "
            "Remove treatment details, induction agents, durations, procedural wording, process descriptions, "
            "and other cell states that are not differentiation status. "
            "Do not assume that treatment, induction, maturation, quiescence, senescence, or apoptosis implies a differentiation state "
            "unless the state is explicitly stated by the current value."
        ),

        "confluence": (
            "Keep only explicit confluence values, confluence ranges, or qualitative confluence states, such as 80%, 80-85%, near confluence, or full confluence. "
            "Normalize decimal confluence fractions to percentage form when clearly supported, such as 0.4 confluence to 40%. "
            "Remove approximation markers and surrounding procedural wording when they do not change the core meaning. "
            "If multiple explicit confluence values are present, return them separately in source order. "
            "Do not keep time durations, cell density measurements such as cells per area, or generic density expressions unless the text explicitly describes them as confluence."
        ),
    },

    "electrochemistry": {
        "electrolytes.chemical": (
            "Normalize to one or more canonical electrolyte-component identities. "
            "Keep each chemically distinct electrolyte constituent as its own value, including salts, acids, bases, solvents, ionic liquids, added water, buffer identities, and additives when explicitly part of the electrolyte composition. "
            "Prefer a single canonical surface form that is sufficient to identify the component. "
            "For obvious surface variants of the same component, use one consistent canonical form, including differences in spacing, hyphenation, bracketed formula formatting, and equivalent formula-style notation. "
            "For values that contain both a full name and an abbreviation, keep a single canonical form rather than the combined string; prefer the abbreviation if it is a stable component identity, otherwise keep the standard full name. "
            "Preserve chemically meaningful distinctions such as different counterions, oxidation states, substituents, solvent identities, and explicitly stated hydrate or complex forms. "
            "Keep established mixture identities such as phosphate buffer or acetate buffer when the current value names the mixture itself. "
            "Remove non-identity details such as concentration, ratio, percentage, pH, purity, and supplier information. "
            "If the value indicates the absence of a component, such as 0% X, X-free, without X, or no X, return []. "
            "Return [] if the value is too generic to identify a specific component."
        ),

        "electrolytes.concentration": (
            "Normalize to one or more canonical concentration or composition values for the electrolyte component. "
            "Keep only explicit concentration, fraction, percentage, loading, or mixture-ratio values attached to the current component. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 0 to 100 mM into hyphenated forms such as 0-100 mM. "
            "Use a unified unit style: convert mol/L, mol L−1, mol dm−3, mmol/L, mmol L−1, μmol/L, umol/L, and µmol L−1 to mM; mg L−1 to mg/L; μg L−1 and ug/L to μg/L. "
            "Normalize mass percent forms such as wt.%, weight %, and w/w % to wt%; volume percent forms such as vol.%, volume %, and v/v % to vol%; and weight/volume forms such as w/v %, g/100 mL, or mg/mL percentages to % w/v when explicitly expressed as a percentage basis. "
            "Keep ppm, ppb, mole fraction, and ratio forms as such, using stable formats such as ppm, ppb, mole fraction 0.2, and 1:1 (v/v). "
            "Remove redundant trailing zeros, unify spacing and symbols, and remove non-concentration details. "
            "Do not infer missing units, missing bases, or converted values that are not explicitly supported. "
            "Return [] if the value does not identify a specific concentration or composition value."
        ),
        
        "reference_electrode": (
            "Normalize to one canonical reference-electrode identity value. "
            "Keep the core reference-electrode identity using a stable standard name, abbreviation, or formula-style form that is sufficient to distinguish the electrode. "
            "Unify obvious surface variants of the same electrode, including spacing, punctuation, slash versus bar notation, and standard abbreviation versus full-name variants when they clearly refer to the same identity. "
            "Preserve distinctions that matter for retrieval, including different reference-electrode families, different electrolyte or filling systems when explicitly part of the identity, and quasi-reference or pseudo-reference status. "
            "Remove non-identity details such as concentration, size, shape, vendor information, leak-free wording, and role descriptions such as combined counter and reference, unless they are needed to identify the electrode itself. "
            "Return [] if the value is too generic to identify a specific reference electrode."
        ),
        
        "working_electrode": (
            "Normalize to one canonical working-electrode identity value. "
            "Keep the core working-electrode identity using a stable material, named sample, electrode-name, or formula-style form that is sufficient to distinguish the electrode. "
            "Unify obvious surface variants of the same electrode, including spacing, punctuation, abbreviation versus full-name variants, singular versus plural forms, and stable support notation such as on versus /. "
            "Preserve distinctions that matter for retrieval, including different electrode materials, alloys, crystal faces when tied to a specific material identity, carbon forms, named substrates, and explicitly stated coatings, modifiers, deposited layers, supports, or electrode configurations such as RDE when they are part of the working-electrode identity. "
            "Remove non-identity details such as concentration, size, vendor information, procedural wording, general role labels, generic morphology-only descriptors, and packaging or insulation details such as glass-isolated or glass capillary sealed unless they are needed to distinguish a stable named electrode identity. "
            "Do not keep overly generic electrode-class labels such as electrode, disk electrode, wire electrode, or microelectrode by themselves unless they are anchored by a specific material, named sample, modifier, support, or other stable identity. "
            "Remove shell words such as electrode when the remaining value already identifies the working electrode clearly. "
            "Return [] if the value is too generic to identify a specific working electrode."
        ),
        
        "target_species": (
            "Normalize to one or more canonical target-species identities. "
            "Keep the explicit analyte, reactant, intermediate, or key electroactive species that is the main target of detection, measurement, oxidation, reduction, or mechanistic study. "
            "Prefer a single canonical surface form that is sufficient to identify the species, using a standard chemical name, formula, or well-established abbreviation. "
            "Unify obvious surface variants of the same species, including name versus formula variants and standard abbreviation versus full-name variants, when they clearly refer to the same identity. "
            "Preserve distinctions that matter for retrieval, including different chemical species, oxidation states, charge states, protonation states, complexes, adsorbed intermediates, and explicitly stated molecular forms. "
            "Remove non-identity details such as concentration, solvent, electrolyte components, pH, catalyst or electrode materials, procedural wording, and signal or performance descriptions. "
            "Do not keep generic role labels or system-level entities such as biofilm, enrichment culture, microorganism, or microbial community unless a specific target species identity is explicitly given and is itself the target being measured or studied. "
            "Return [] if the value is too generic to identify a specific target species."
        ),
        
        "temperature": (
            "Normalize to one or more canonical experimental-temperature values. "
            "Keep only explicitly stated thermal conditions, such as 25 °C, 80 °C, room temperature, or 298 K. "
            "Unify obvious surface variants of the same temperature expression, including spacing, degree-symbol variants, and Celsius or Kelvin formatting. "
            "Preserve distinctions that matter for retrieval, including different numeric temperatures, ranges, and qualitative temperature labels such as room temperature when no numeric value is explicitly given. "
            "Keep ranges as single range expressions and normalize textual range forms such as 20 to 30 °C into hyphenated forms such as 20-30 °C. "
            "Convert Kelvin values to °C only when the conversion is exact and straightforward; otherwise keep the original unit. "
            "Remove non-temperature details such as heating time, drying or annealing atmosphere, pressure, humidity, water content, electrolyte-conditioning descriptions, and other non-thermal conditions. "
            "Return [] if the value does not identify a specific temperature condition."
        ),
        
        "measurement_method": (
            "Normalize to one or more canonical electrochemical-measurement method values. "
            "Keep only the explicit measurement or electrochemical technique itself, such as cyclic voltammetry, chronoamperometry, electrochemical impedance spectroscopy, differential pulse voltammetry, square wave voltammetry, linear sweep voltammetry, or chronopotentiometry. "
            "Prefer a well-established abbreviation as the canonical form when it is standard and unambiguous, and do not return combined labels such as full name plus abbreviation."
            "Preserve distinctions that matter for retrieval, including different measurement families and explicitly stated method variants when they represent different techniques. "
            "Remove non-method details such as scan rate, potential window, frequency range, electrode setup, cell configuration, software or instrument names, and procedural wording. "
            "Do not keep generic labels such as electrochemical measurement, electrochemical test, voltammetric method, or electrochemical technique unless a specific method identity is given. "
            "Return [] if the value is too generic to identify a specific measurement method."
        ),
        
        "scan_rate": (
            "Normalize to one or more canonical scan-rate values. "
            "Keep only explicit scan-rate values used in voltammetric measurements, such as 50 mV/s, 0.1 V/s, or 10-100 mV/s. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 10 to 100 mV/s into hyphenated forms such as 10-100 mV/s. "
            "Use a unified unit style: convert V s−1, V/s, mV s−1, mV/s, and mV sec−1 to mV/s. "
            "Keep one stable numeric form, remove redundant trailing zeros, and unify spacing and symbols. "
            "Remove non-scan-rate details such as potential windows, step potentials, pulse amplitudes, cycle numbers, frequency, electrode information, and procedural wording. "
            "Do not infer missing units or converted values that are not explicitly supported. "
            "Return [] if the value does not identify a specific scan-rate condition."
        ),
        
        "applied_potential": (
            "Normalize to one or more canonical applied-potential values. "
            "Keep only explicit fixed potentials, potential ranges, or potential windows used in the electrochemical measurement. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions, normalize textual range forms into range form, and order endpoints from lower to higher numeric value. "
            "Format ranges with spaces around the separator, such as -0.4 - 0 V or 0.2 - 1.0 V. "
            "Preserve the reference basis when explicitly stated, using stable forms such as vs Ag/AgCl, vs SCE, vs SSCE, vs SHE, vs RHE, or vs Li/Li+. "
            "Use a unified unit style, remove redundant trailing zeros, and unify spacing and symbols. "
            "Remove non-potential details such as scan rate, pulse amplitude, frequency, cycle number, sample labels, and procedural wording. "
            "Return [] if the value does not identify a specific applied-potential condition."
        ),

    },

    "immunotechnology": {
        "protein_targets.target": (
            "Keep only the core protein target identity intentionally probed in the blot. "
            "Remove antibody wording, product details, vendor information, and other reagent description. "
            "Remove form/state information such as phospho, phosphorylated, p-, total, cleaved, mutation-site notation, and residue-site notation. "
            "Keep subunit, isoform, or family-member identifiers only when they are part of the target identity itself. "
            "When a value contains both a longer name and a shorter symbol for the same target, keep one concise canonical form. "
            "If an epitope or motif expression clearly refers to a named protein target, keep only the core protein target identity; otherwise return []. "
            "If a slash-form value explicitly contains more than one true target identity, you may split them into separate values. "
            "Do not keep loading or normalization controls here; if the evidence clearly assigns a control role, return []. "
            "If the value is only a generic reagent label, IgG/isotype control, or secondary antibody, return []."
        ),

        "protein_targets.target_form": (
            "Keep only the target form or state when it is explicitly stated or directly supported by the current value. "
            "Use concise canonical labels such as phospho, total, cleaved, precursor, mature, or activated only when they clearly belong to the probed target form. "
            "Remove target identity, antibody wording, residue-site notation, mutation-site notation, species names, vendor information, and other reagent or product description. "
            "Map obvious surface variants to one canonical form, such as p-, phospho-, or phosphorylated to phospho. "
            "If the form is not clearly stated, return []. "
            "Do not introduce a form from nearby evidence if it is not already grounded by the current value."
        ),

        "normalization_control": (
            "Keep only the core protein identity explicitly used as a loading control or normalization control in the blot. "
            "Remove control-role wording, antibody wording, and other reagent or product description. "
            "Remove staining methods, total-protein references, molecular-weight band descriptions, dyes, chemical reagents, and other non-protein normalization signals; return [] for those cases. "
            "Normalize only obvious surface variation, such as beta-actin and β-actin to one consistent form. "
            "Keep subtype information only when it is part of the control identity itself. "
            "Do not keep modification or target-form wording such as acetylated, phospho, or cleaved unless the remaining core protein identity can be cleanly kept. "
            "If the value does not contain an actual control protein identity and only gives a generic control label, return []. "
            "If a value explicitly contains more than one true control protein identity, you may split them."
        ),

        "sample_type": (
            "Keep only the core sample category used for blotting. "
            "Preserve meaningful distinctions such as cell lysate, tissue lysate, total protein extract, nuclear extract, cytoplasmic extract, membrane fraction, mitochondrial fraction, serum, and plasma. "
            "Merge only obvious near-equivalent surface variants within the same sample category, but do not merge distinct categories that still matter for retrieval. "
            "Remove cell line names, tissue names, species names, disease labels, treatment groups, cell-state labels, and culture-mode descriptors such as 2D, 3D, spheroid, organoid, or co-culture. "
            "Remove buffer names, extraction reagents, preparation steps, procedural wording, centrifugation details, and other surrounding description. "
            "When a value contains two near-equivalent descriptions of the same sample, keep one stable canonical category rather than splitting them into multiple values. "
            "If the value does not contain an actual sample category, return []."
        ),

        "lysis_buffer": (
            "Keep only the core identity of the lysis or extraction buffer used to prepare the sample. "
            "Preserve stable named or pattern-stable buffer identities such as RIPA buffer, NP-40 lysis buffer, urea buffer, Laemmli buffer, SDS lysis buffer, or similar lysis or extraction buffers. "
            "When an unnamed buffer is expressed mainly as a composition, keep it only if the composition clearly indicates a stable and reusable buffer identity; otherwise return []. "
            "Remove full composition details, concentrations, pH values, additives, inhibitors, reducing agents, salts, and other recipe-level description unless they are needed to determine the core buffer identity. "
            "Remove procedural wording, incubation conditions, vendor information, catalogue numbers, and other surrounding description. "
            "Do not keep sample-type expressions such as cell lysate, tissue lysate, nuclear extract, serum, or plasma. "
            "When a value contains both a recipe and a clearer buffer identity, keep one stable canonical form. "
            "If the value does not contain an actual lysis or extraction buffer identity, return []."
        ),

        "blocking_solution.identity": (
            "Keep only the core blocking reagent identity or named blocking solution used before antibody incubation. "
            "Preserve meaningful identities such as BSA, bovine serum albumin, nonfat dry milk, skim milk, casein, normal serum, or named blocking products such as Intercept TBS Blocking Buffer. "
            "Remove concentration, percentage, dilution, buffer context such as TBST or PBS, incubation conditions, procedural wording, vendor information, catalogue numbers, and other surrounding description unless they are part of the fixed name of a named blocking solution itself. "
            "When a value contains both a generic description and a clearer blocking-solution identity, keep one stable canonical form. "
            "Do not keep detergents, wash buffers, antibody diluents, or other reagents that are not themselves the blocking reagent or blocking solution; return [] for those cases. "
            "If the value does not contain an actual blocking reagent identity, return []."
        ),

        "blocking_solution.concentration": (
            "Keep only the explicit concentration, percentage, dilution, or working-strength expression of the blocking solution, such as 5%, 3%, 1%, 1:10, or 1×. "
            "Remove blocking-reagent identity, product names, buffer context such as TBST or PBS, incubation conditions, procedural wording, vendor information, catalogue numbers, and other surrounding description. "
            "Normalize only obvious surface variation, such as 1X to 1×, and keep one concise canonical concentration form. "
            "If the value does not contain an actual concentration, dilution, or working-strength expression for the blocking solution, return []. "
            "If a value explicitly contains more than one concentration expression for the same blocking solution, you may keep the clearly relevant one or split them only when they are truly separate blocking concentrations."
        ),

        "gel_percentage": (
            "Keep only the explicit SDS-PAGE gel percentage or percentage range, such as 8%, 10%, 12%, or 10-12%. "
            "Remove gel type wording, procedural wording, electrophoresis context, vendor information, and other surrounding description. "
            "Normalize only obvious surface variation, such as 10 % to 10%, and use a consistent form for ranges such as 10-12%. "
            "If the value does not contain an actual gel percentage expression, return []. "
            "If a value contains more than one distinct gel percentage used as separate gel conditions, you may split them."
        ),

        "loading_amount": (
            "Keep only the explicit sample or protein loading amount per lane or well, such as 20 ug, 30 μg, 50 ug/lane, or 10 μl/well. "
            "Remove sample identity, protein identity, gel context, lane or well wording when it is not needed, procedural wording, incubation or electrophoresis details, vendor information, and other surrounding description. "
            "Preserve the amount together with its unit, and keep per-lane or per-well wording only when it is explicitly present and helps define the loading amount. "
            "Normalize only obvious surface variation, such as ug to μg and spacing differences, and keep one concise canonical form. "
            "If the value does not contain an actual loading amount expression, return []. "
            "If a value explicitly contains more than one distinct loading amount used as separate loading conditions, you may split them."
        ),  
    },

    "inorganic_materials": {
        "target_material": (
            "Keep only the core target inorganic material identity or material system represented by the current value. "
            "Preserve distinctions that change the material system itself, such as different compounds, phases, composites, solid solutions, or explicitly stated doped/substituted material identities. "
            "Use one clean canonical material name or formula when the current value already supports it. "
            "Remove processing details, precursor information, additive information, morphology-only description, sample labels, supplier information, particle size, purity, and result-oriented wording. "
            "Remove standalone amounts, percentages, and composition ratios when they are only quantitative modifiers rather than the core material identity. "
            "Do not rewrite a specific material system into a broader base material. "
            "Do not introduce precursor, additive, or neighboring material names from the evidence if they are not part of the current value itself. "
            "If the value is too generic to identify a specific target material, return []."
        ),

        "precursors": (
            "Keep one or more canonical precursor-material identities explicitly represented by the current value. "
            "Treat this field as starting-material identity rather than exact formulation, so remove amounts, ratios, concentrations, and other quantitative modifiers that do not change the core precursor identity. "
            "Preserve distinctions that change the precursor itself, such as different compounds, salts, oxides, carbonates, nitrates, acetates, and hydrated precursor identities when hydration is part of the named chemical identity. "
            "Use one clean canonical material name or formula, and when a value contains both a formula and a full name for the same precursor, keep only one canonical form. "
            "Split multiple true precursor identities only when they are explicitly contained in the current value itself. "
            "Normalize obvious surface variation in formula formatting, spacing, hyphenation, and separator style, but avoid special typographic variants that reduce matching stability. "
            "Remove procedural wording, target-material wording, role labels, supplier information, purity, particle size, and other non-identity details. "
            "Do not rewrite the value into a broader material class, and do not add new precursor names from the evidence. "
            "If the value is too generic to identify a specific precursor material, return []."
        ),

        "dopant_or_additive.identity": (
            "Keep one or more canonical dopant or additive identities explicitly represented by the current value. "
            "Use this field only for meaningful dopants, sintering aids, or processing additives introduced into the material system. "
            "Preserve distinctions that change the additive identity itself, and use one clean canonical material name or formula. "
            "Split multiple true additive identities only when they are explicitly contained in the current value itself. "
            "Remove amounts, ratios, concentrations, role wording, procedural wording, and other non-identity details. "
            "Do not keep vague labels without a specific chemical identity, and do not keep partial composition fragments, variable expressions, or incomplete identities extracted from a larger material formula. "
            "Do not add new additive names from the evidence. "
            "If the value is too generic to identify a specific dopant or additive, return []."
        ),

        "dopant_or_additive.amount": (
            "Normalize to one or more canonical dopant/additive amount values. "
            "Keep only explicit amount expressions for dopants or additives, such as 1 wt%, 5 mol%, 0.5 at%, 10 vol%, 500 ppm, molar ratios, or loading amounts. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 0 to 5 wt.% into hyphenated forms such as 0-5 wt%. "
            "Use a unified unit style, including wt%, mol%, at%, vol%, and ppm, and normalize obvious surface variation in spacing, punctuation, and percent formatting. "
            "Keep one stable numeric form and remove redundant trailing zeros. "
            "Remove dopant/additive identity, role wording, procedural wording, and other non-amount details. "
            "Do not infer missing units or add new amount values from the evidence. "
            "Return [] if the value does not identify a specific amount condition."
        ),
        
        "synthesis_route": (
            "Keep only the core synthesis, consolidation, or named material-processing route represented by the current value. "
            "Preserve distinctions that change the route itself, and use one concise canonical route name. "
            "Keep named routes such as sol-gel, hydrothermal synthesis, spark plasma sintering, 3D printing, acid leaching, acid treatment, functionalization, or carbonation when they are presented as meaningful preparation or processing methods. "
            "Remove temperature, time, atmosphere, pressure, heating rate, equipment details, sample form, and other non-route process details. "
            "For multi-step expressions, keep only the main named route or routes explicitly represented by the current value, but do not split off generic downstream steps such as drying, debinding, or sintering unless they are themselves the core route identity. "
            "Do not broaden a specific route or add a new route from the evidence. "
            "If the value is too generic to identify a specific route, return []."
        ),
        
        "atmosphere": (
            "Keep one or more canonical atmosphere-component identities explicitly represented by the current value. "
            "Preserve atmosphere identities such as O2, N2, Ar, H2, air, vacuum, inert gas, or reducing atmosphere, and split explicitly stated mixed-gas atmospheres into separate component values when appropriate. "
            "Prefer stable chemical-form expressions such as O2, N2, Ar, and H2 when applicable. "
            "Remove temperature, time, pressure, flow rate, dew point, equipment details, and other non-atmosphere details. "
            "Do not infer an atmosphere identity from pressure alone, and do not add a new atmosphere from the evidence. "
            "If the value is too generic to identify a specific atmosphere, return []."
        ),
        
        "heating_rate": (
            "Normalize to one or more canonical heating-rate values. "
            "Keep only explicit heating-rate values used in thermal processing, such as 5 °C/min, 10 °C/min, or 2-10 °C/min. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 2 to 10 °C/min into hyphenated forms such as 2-10 °C/min. "
            "Use a unified unit style: convert K/min, K/s, °C/s, °C min−1, and other equivalent forms to °C/min. "
            "When multiple equivalent forms are given for the same heating rate, keep a single canonical °C/min value. "
            "Keep one stable numeric form, remove redundant trailing zeros, and unify spacing and symbols. "
            "Remove temperature, time, atmosphere, pressure, and other non-heating-rate process details. "
            "Do not infer missing units or add new values from the evidence. "
            "Return [] if the value does not identify a specific heating-rate condition."
        ),

        "pressing_pressure": (
            "Normalize to one or more canonical pressing-pressure values. "
            "Keep only explicit pressure values applied to the sample during shaping, pressing, compaction, hot pressing, or related consolidation steps. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 10 to 30 MPa into hyphenated forms such as 10-30 MPa. "
            "Use a unified unit style: convert equivalent pressure-unit forms such as GPa, kbar, ton/cm2, t/cm2, and kgf/cm2 to MPa when explicitly supported. "
            "Keep one stable numeric form, remove redundant trailing zeros, and unify spacing and symbols. "
            "Remove temperature, time, atmosphere, gas pressure, suction pressure, chamber pressure, vacuum pressure, equipment details, and other non-pressing-pressure details. "
            "Do not keep force or load values such as ton, t, tf, kg, or kN unless they are explicitly expressed per unit area as pressure. "
            "Do not infer missing units or add new values from the evidence. "
            "Return [] if the value does not identify a specific pressing-pressure condition."
        ),
        
        "sintering.temperature": (
            "Normalize to one or more canonical sintering-temperature values. "
            "Keep only explicit sintering temperatures used for the main sintering step. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual range forms such as 1000 to 1200 °C into hyphenated forms such as 1000-1200 °C. "
            "Use a unified unit style: convert equivalent temperature-unit forms such as K to °C when explicitly supported. "
            "Keep one stable numeric form, remove redundant trailing zeros, and unify spacing and symbols. "
            "Remove sintering time, heating rate, atmosphere, pressure, ramp details, and other non-temperature process details. "
            "Do not keep temperatures for drying, calcination, annealing, debinding, melting, or other non-sintering steps. "
            "Do not infer missing units or add new values from the evidence. "
            "Return [] if the value does not identify a specific sintering-temperature condition."
        ),
        
        "sintering.time": (
            "Normalize to one or more canonical sintering-time values. "
            "Keep only explicit dwell or holding times for the main sintering step. "
            "Use h as the unified unit and convert equivalent forms such as s, min, hr, and hour(s) to h when explicitly supported. "
            "Split explicitly listed series values into separate canonical values and keep ranges as single normalized range expressions. "
            "Convert mixed expressions such as 1 h and 30 min to a single h value such as 1.5 h. "
            "Remove temperature and other non-time details, do not keep values without an explicit time unit, and do not add new values from the evidence. "
            "Return [] if the value does not identify a specific sintering-time condition."
        ),
        
        "post_treatment.type": (
            "Keep only the core post-treatment type represented by the current value. "
            "Use this field for distinct post-treatment identities such as annealing, calcination, oxidation, reduction, quenching, or heat treatment when they are meaningful processing steps separate from the main sintering step. "
            "Use one concise canonical treatment name. "
            "Remove temperature, time, atmosphere, pressure, equipment details, purpose wording, and other non-type details. "
            "Do not keep routine drying, cooling, storage, or mechanical surface treatment unless the value clearly presents them as meaningful post-treatment steps. "
            "If multiple expressions in the same value refer to the same treatment step, keep only one canonical type. "
            "Do not add a new treatment type from the evidence. "
            "If the value is too generic to identify a specific post-treatment type, return []."
        ),
        
        "post_treatment.temperature": (
            "Normalize to one or more canonical post-treatment temperature values. "
            "Keep only explicit temperatures for distinct post-treatment steps such as annealing, calcination, oxidation, reduction, or other meaningful thermal treatments separate from the main sintering step. "
            "Split explicitly listed series values into separate canonical values and keep ranges as single normalized range expressions. "
            "Use °C as the unified unit and convert equivalent forms such as K to °C when explicitly supported. "
            "Normalize spacing and symbols, keep one stable numeric form, and remove redundant trailing zeros. "
            "Remove time, atmosphere, pressure, equipment details, and other non-temperature details. "
            "Do not keep temperatures from drying, sintering, debinding, or other non-post-treatment steps. "
            "Do not infer missing units or add new values from the evidence. "
            "Return [] if the value does not identify a specific post-treatment temperature condition."
        ),
        
        "post_treatment.time": (
            "Normalize to one or more canonical post-treatment time values. "
            "Keep only explicit times for distinct thermal post-treatment steps separate from the main sintering step. "
            "Use h as the unified unit and convert equivalent forms such as s, min, hr, and hour(s) to h when explicitly supported. "
            "Split explicitly listed series values into separate canonical values and keep ranges as single normalized range expressions. "
            "Convert mixed expressions such as 1 h and 30 min to a single h value such as 1.5 h. "
            "Remove temperature and other non-time details, do not keep times from non-post-treatment steps, and do not add new values from the evidence. "
            "Return [] if the value does not identify a specific post-treatment time condition."
        ),
    },

    "organic_synthesis": {
        "transformation_type": (
            "Keep only the reaction class or transformation type expressed by the current value. "
            "Retain explicit named reactions, stable transformation labels, and broad but meaningful reaction-class labels, "
            "such as Suzuki coupling, Sonogashira coupling, hydrogenation, oxidation, reduction, cyclization, cycloaddition, amidation, arylation, borylation, or difunctionalization. "
            "Remove substrate names, product names, reaction partners, catalyst or reagent information, conditions, procedural wording, and result-oriented language. "
            "When the current value contains both a broad transformation label and a more specific subtype for the same reaction, keep the more specific stable label rather than both. "
            "Do not keep mechanistic step labels or mechanistic commentary, such as oxidative addition, reductive elimination, migratory insertion, or related phrases, unless the value itself clearly uses them as the transformation label. "
            "If a value only describes conversion of a substrate without clearly expressing a transformation type, return []. "
            "If a conversion expression explicitly includes more than one independent transformation type, return those transformation types as separate normalized values. "
            "Normalize obvious formatting variants to one consistent canonical form, including bracket style and spacing in expressions such as [2+1], [2+3], or [3+2]."
        ),
        
        "substrates": (
            "Keep only the substrate or reaction-partner identity represented by the current value. "
            "Retain specific chemical identities or stable substrate-class labels when they are explicitly present. "
            "Normalize obvious surface variants, but do not infer a more specific identity from the evidence. "
            "Remove amounts, equivalents, concentrations, supplier information, purity, physical form, procedural wording, product-oriented wording, and local labels such as 1a or compound 5 when they do not express chemical identity. "
            "Do not keep catalysts, ligands, bases, additives, solvents, products, intermediates, or other non-substrate roles. "
            "Do not keep stoichiometric reagents, chiral reagents, reducing agents, oxidants, or other named reaction reagents unless the current value clearly denotes a substrate or reaction partner being transformed."
            "If the value does not clearly denote a substrate or reaction-partner identity, return []. "
            "If more than one independent substrate or reaction partner is explicitly present in the current value, return them as separate normalized values."
        ),
        
        "catalysts.name": (
            "Keep only the catalyst identity in the current value. "
            "Retain specific catalyst names, formulas, or stable catalyst labels, and normalize obvious surface variants. "
            "Remove loading, concentration, equivalents, procedural wording, supplier details, purity, physical form, support amount, and local labels that do not express catalyst identity. "
            "Do not keep free ligands, bases, additives, solvents, substrates, products, intermediates, or other non-catalyst roles. "
            "Do not infer a fuller catalyst identity from the evidence. "
            "If the current value is only a vague role word or does not clearly denote a catalyst identity, return []. "
            "If multiple independent catalysts are explicitly present, return them separately."
        ),
        
        "catalysts.loading": (
            "Keep only the explicitly stated catalyst loading in the current value. "
            "Retain loading expressions such as mol %, equiv, mol, mmol, or μmol when explicitly given. "
            "Normalize obvious surface variants in spacing, decimals, and unit writing. "
            "Use consistent unit forms: mol%, mol %, mole %, and mole% → mol %; equivalent, equiv., and eq. → equiv; umol and µmol → μmol; mmole and mmoles → mmol. "
            "Only unify surface forms within the same unit type, and do not force all values into one common unit. "
            "Remove catalyst identity and other details that do not belong to catalyst loading. "
            "Do not infer or calculate new values from the evidence. "
            "If the value does not clearly denote catalyst loading, return []. "
            "If multiple independent loading values are explicitly present, return them separately."
        ),
        
        "ligands": (
            "Keep only the ligand identity in the current value. "
            "Retain ligand names, formulas, abbreviations, or stable ligand labels, and normalize obvious surface variants. "
            "If a ligand is given together with a local paper-specific label such as L1 or L10, keep only the ligand identity and remove the local label. "
            "Remove metal names, catalyst descriptions, loading, procedural wording, and other details that do not belong to ligand identity. "
            "Do not keep catalysts, bases, additives, solvents, substrates, products, or other non-ligand roles. "
            "Do not infer new ligand identities from the evidence. "
            "If the value is only a local paper-specific code or does not clearly denote a ligand identity, return []. "
            "If multiple independent ligands are explicitly present, return them separately."
        ),
        
        "bases": (
            "Keep only the base identity in the current value. "
            "Retain base names, formulas, abbreviations, or stable base labels, and normalize obvious surface variants. "
            "Use a consistent canonical form: prefer widely established abbreviations such as DBU or DIPEA when they are standard and unambiguous; otherwise keep a minimally cleaned base identity rather than rewriting it into a fuller name. "
            "Remove amounts, concentration, equivalents, procedural wording, supplier details, purity, physical form, and local labels that do not express base identity. "
            "Do not keep catalysts, ligands, additives, solvents, substrates, products, or other non-base roles. "
            "Do not infer new base identities from the evidence. "
            "If the value does not clearly denote a base identity, return []. "
            "If multiple independent bases are explicitly present, return them separately."
        ),
        
        "additives": (
            "Keep only the identity of a functional additive or promoter in the current value. "
            "Retain additive names, formulas, abbreviations, or stable additive labels, and normalize obvious surface variants. "
            "When a full name is paired with an abbreviation, keep only one stable canonical form; keep the abbreviation only if it is widely established and unambiguous, otherwise keep the fuller identity. "
            "Remove amounts, concentration, equivalents, procedural wording, supplier details, purity, physical form, and local labels that do not express additive identity. "
            "Do not keep catalysts, ligands, bases, solvents, substrates, products, or other non-additive roles. "
            "Do not keep titration reagents, workup reagents, or general stoichiometric reaction reagents unless the current value clearly denotes a functional additive used in the reaction itself. "
            "Do not infer new additive identities from the evidence. "
            "If the value does not clearly denote an additive identity, return []. "
            "If multiple independent additives are explicitly present, return them separately."
        ),
        
        "solvents": (
            "Keep only the solvent identity in the current value. "
            "Retain solvent names, formulas, abbreviations, or stable solvent labels, and normalize obvious surface variants. "
            "When a solvent has a widely used and unambiguous canonical form, keep one stable form rather than multiple equivalent expressions. "
            "Remove volume, concentration, ratio, procedural wording, supplier details, purity, drying status, degassing status, and other details that do not belong to solvent identity itself. "
            "Do not keep catalysts, ligands, bases, additives, substrates, products, electrolytes, anolytes, catholytes, or other non-solvent roles. "
            "Do not infer new solvent identities from the evidence, including inferring water from aqueous, aq, or solution descriptions. "
            "If the value is only an electrolyte or solution description and does not clearly denote a solvent identity, return []. "
            "If multiple independent solvent components are explicitly present, return them separately."
        ),
        
        "activation_mode": (
            "Keep only the reaction activation mode in the current value. "
            "Normalize to one or more of these canonical labels when clearly supported: thermal, photochemical, electrochemical, microwave, mechanochemical, sonochemical, or plasma. "
            "Use the current value as the anchor and use the evidence only to clarify the same value. "
            "Map obvious variants such as heating, reflux, annealing, calcination, or elevated temperature to thermal; light irradiation or LED to photochemical; electrolysis, applied current, or applied potential to electrochemical; microwave irradiation to microwave; ball milling or grinding to mechanochemical; ultrasound or sonication to sonochemical; and DBD plasma or plasma discharge to plasma. "
            "Remove atmosphere, flow mode, reactor type, pressure, catalyst, reagent, solvent, temperature, time, device details, and other wording that does not belong to activation mode itself. "
            "Do not infer an activation mode from weak or indirect context. "
            "If the value does not clearly denote an activation mode, return []. "
            "If multiple independent activation modes are explicitly present, return them separately."
        ),
    },

    "polymer_materials": {
        "polymer_system": (
            "Keep only the core identity of the main photocurable polymer, resin, monomer, oligomer, or crosslinkable network system represented by the current value. "
            "If the current value explicitly contains multiple main polymer-system components, split them into separate normalized values. "
            "Use one consistent canonical form per value. Prefer a stable concise identifier already supported by the current value, such as GelMA, PEGDA, HDDA, UDMA, or HPMA; otherwise keep a minimally cleaned full identity. "
            "Do not keep both a full name and its abbreviation together as one value. Do not prefer document-local shorthand for generic formulation phrases. "
            "Remove concentration, ratio, molecular-weight grade, supplier information, catalog details, procedural wording, and generic words such as blend, hydrogel, resin, ink, precursor, or solution when they do not change the core identity. "
            "Do not keep photoinitiators, crosslinking agents, solvents, cells, drugs, fillers, nanoparticles, proteins, peptides, or other additives unless they are clearly part of the main polymer system itself. "
            "If the current value is only a generic formulation phrase without a usable polymer identity, or explicitly indicates no polymer system, return []."
        ),
        
        "polymer_concentration": (
            "Keep explicitly stated concentration, fraction, loading, or amount-per-volume values represented by the current value when they can serve as usable concentration-style constraints for the main polymer system. "
            "High recall is preferred for this field. If the current value explicitly contains multiple concentration values, including values for multiple co-formulated polymer-system components, keep all usable concentration values as separate normalized values. "
            "Normalize obvious unit-surface variation to a consistent form, such as wt.% to wt%, %, w/v to % w/v, %, v/v to % v/v, and %, w/w to % w/w. "
            "When the current value gives a directly computable amount-in-volume expression, such as x g in y mL, convert it to a consistent concentration form such as mg/mL when the conversion is straightforward and unambiguous. "
            "You may also keep standard concentration forms such as wt%, % w/v, % v/v, % w/w, mg/mL, g/L, or M. "
            "Remove polymer names, supplier information, procedural wording, and surrounding formulation text when they are not part of the concentration value itself. "
            "If the value clearly belongs to a solvent, photoinitiator, crosslinking agent, or another non-polymer component rather than the main polymer system, return []. "
            "Do not infer missing units or perform conversions that require assumptions beyond the explicit value."
        ),
        
        "photoinitiators.name": (
            "Keep only the identity of the photoinitiator represented by the current value. "
            "High recall is preferred for this field. If the current value explicitly contains multiple photoinitiator identities, keep them as separate normalized values. "
            "Use one consistent canonical form per value. Prefer a minimally cleaned full identity by default. "
            "Use a concise identifier only when it is clearly a stable and widely used photoinitiator name already supported by the current value, such as LAP, I2959, TPO, BAPO, eosin Y, or camphorquinone. "
            "Do not keep both a full name and its abbreviation together as one value. Do not prefer document-local shorthand, product names, or catalog-like names as the canonical value. "
            "Do not keep bare numeric codes or fragmentary labels, such as 184 or VA-like shorthand, unless they are self-contained and stable photoinitiator identities. When a value is an obvious spelling or spacing variant of an already established photoinitiator identity, normalize it to that same canonical form."
            "Remove loading, concentration, supplier information, catalog details, procedural wording, solvent wording, and other surrounding formulation text when they are not part of the initiator identity itself. "
            "Do not keep polymers, crosslinking agents, co-monomers, solvents, fillers, nanoparticles, light-source descriptions, or generic phrases such as photoinitiator system when no specific initiator identity is given. "
            "If the current value does not provide a usable photoinitiator identity, is only a fragmentary label, or explicitly indicates the absence of a photoinitiator, return []."
        ),
        
        "photoinitiators.loading": (
            "Normalize to one or more canonical photoinitiator-loading values. "
            "Keep only explicit loading or concentration values for photoinitiators, such as 0.5 wt%, 1 mol%, 0.1% w/v, 5 mg/mL, 0.01 mol/10 mL, or 2-10 wt%. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual forms such as 0.01 to 0.1% w/v to 0.01-0.1% w/v. "
            "Normalize obvious unit-surface variation, such as wt.% to wt%, mol % to mol%, %, w/v to % w/v, %, w/w to % w/w, and mole to mol. "
            "For amount-per-volume expressions, keep a compact canonical form such as 0.01 mol/10 mL and remove solvent wording. "
            "Remove names, supplier information, procedural wording, and other non-loading details. "
            "Return [] if the value belongs to a co-initiator, crosslinking agent, polymer, solvent, filler, or another non-photoinitiator component. "
            "Do not infer missing units or perform conversions that require extra assumptions."
        ),

        "crosslinking_agent.name": (
            "Keep only the identity of explicitly added crosslinking agents or other distinct reactive formulation additives represented by the current value. "
            "High recall is preferred for this field. If the current value explicitly contains multiple usable identities, keep them as separate normalized values. "
            "Use one consistent canonical form per value. Prefer a stable concise identifier already supported by the current value; otherwise keep a minimally cleaned full identity. "
            "Do not keep both a full name and its abbreviation together as one value. "
            "Remove loading, concentration, supplier information, catalog details, procedural wording, and other surrounding formulation text. "
            "Do not keep the main polymer system, photoinitiators, solvents, fillers, nanoparticles, light-source descriptions, or generic phrases such as crosslinker system when no specific identity is given. "
            "Return [] for co-initiators, inhibitors, catalysts, or other additives that are not clearly used as crosslinking agents or distinct reactive crosslinking-related formulation components."
        ),
        
        "crosslinking_agent.loading": (
            "Normalize to one or more canonical crosslinking-agent-loading values. "
            "Keep only explicit loading or concentration values for crosslinking agents or other retained reactive crosslinking-related additives, such as 0.5 wt%, 1 mol%, 0.1% w/v, 5 mg/mL, 0.01 mol/10 mL, or 2-10 wt%. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual forms such as 0.01 to 0.1% w/v to 0.01-0.1% w/v. "
            "Normalize obvious unit-surface variation, such as wt.% to wt%, mol % to mol%, %, w/v to % w/v, %, w/w to % w/w, and mole to mol. "
            "For amount-per-volume expressions, keep a compact canonical form such as 0.01 mol/10 mL and remove solvent wording. "
            "Remove names, supplier information, procedural wording, and other non-loading details. "
            "Return [] if the value belongs to the main polymer system, a photoinitiator, a co-initiator, a solvent, a filler, or another non-crosslinking component. "
            "Do not infer missing units or perform conversions that require extra assumptions."
        ),

        "wavelength": (
            "Normalize to one or more canonical wavelength values. "
            "Keep only explicit irradiation wavelength values for the curing setup, such as 365 nm, 405 nm, or 395-400 nm. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual forms such as 395 to 400 nm to 395-400 nm. "
            "Use a unified unit style and keep wavelength values in nm. "
            "Keep one stable numeric form, remove redundant trailing zeros when appropriate, and unify spacing and symbols. "
            "Remove light-source names, color descriptions, intensity, exposure time, procedural wording, and other non-wavelength details. "
            "Return [] if the value does not identify a specific wavelength, or only gives vague descriptions such as UV, visible, blue light, or near-IR without an explicit wavelength value. "
            "Do not infer missing wavelengths from the evidence."
        ),
        
        "intensity": (
            "Normalize to one or more canonical light-intensity values. "
            "Keep only explicit irradiation intensity or irradiance values for the curing setup. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual forms such as 1 to 5 mW/cm^2 to 1-5 mW/cm^2. "
            "Convert all directly convertible units to mW/cm^2, including W/cm^2, mW/mm^2, and W/m^2. "
            "Keep one stable numeric form, remove redundant trailing zeros when appropriate, and unify spacing and symbols. "
            "Remove wavelength, exposure time, light-source names, color descriptions, and other non-intensity details. "
            "Return [] if the value does not identify a specific light intensity. "
            "Do not infer missing units or perform conversions that require extra assumptions."
        ),
        
        "exposure_time": (
            "Normalize to one or more canonical exposure-time values. "
            "Keep only explicit irradiation or exposure-time values for the curing setup. "
            "Split explicitly listed series values into separate canonical values and preserve their original order. "
            "Keep ranges as single range expressions and normalize textual forms such as 10 to 30 s to 0.1667-0.5 min. "
            "Convert all directly convertible time units to min. "
            "Keep one stable numeric form, remove redundant trailing zeros when appropriate, and unify spacing and symbols. "
            "Remove wavelength, intensity, light-source names, procedural wording, and other non-time details. "
            "Return [] if the value does not identify a specific exposure time. "
            "Do not infer missing units or perform conversions that require extra assumptions."
        ),
        
        "light_source_type": (
            "Keep only the identity of the explicit light source used for curing, such as LED, UV LED, laser, projector, mercury lamp, xenon lamp, halogen lamp, or UV lamp. "
            "If the current value explicitly contains multiple usable light-source identities, keep them as separate normalized values. "
            "Use one consistent canonical form per value and prefer the most specific source identity already supported by the current value. "
            "Remove wavelength, intensity, exposure time, color wording, supplier information, model names, and other surrounding details that are not part of the source identity itself. "
            "Do not keep generic phrases such as light, irradiation, or photo-curing when no specific source type is given. "
            "Return [] if the value does not provide a usable light-source identity."
        ),
    },

}



def render_prompt(domain, slot_name, batch_items):
    slot_requirement = FIELD_REQUIREMENTS[domain][slot_name]

    prompt_items = []
    for item in batch_items:
        prompt_items.append({
            "item_id": item["item_id"],
            "current_values": item["current_values"],
            "evidence": item["evidence"],
        })

    prompt_text = PROMPT_TEMPLATE.format(
        domain=domain,
        slot_name=slot_name,
        slot_requirement=slot_requirement,
        batch_items=json.dumps(prompt_items, ensure_ascii=False, indent=2)
    )

    system_prompt, user_prompt = prompt_text.split("|split|")
    return system_prompt, user_prompt


if __name__ == "__main__":
    example_prompt = render_prompt(
        domain="electrochemistry",
        slot_name="electrolytes.chemical",
        slot_definition="The electrolyte chemical identity only. Concentration belongs to a separate field.",
        raw_values=[
            "0.1 M KOH",
            "1 M NaOH",
            "O2-saturated 0.1 M KOH",
            "PBS buffer",
            "conductivity measurements"
        ]
    )

    print(example_prompt)


# python scripts/structured_extraction/llm_clean_prompt.py
