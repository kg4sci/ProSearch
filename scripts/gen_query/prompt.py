sys_prompt = """
You are an experienced researcher in {domain}. You understand experimental design, protocol details, and how researchers phrase literature-search requests when they are trying to find concrete experimental evidence in papers.
"""

gen_prompt_reproduction = """You are given a structured experimental record from {domain}. Your task is to decide whether this record should be kept as a final retrieval query candidate, and if it should, to rewrite it as one concise literature-search request.

Read the record as describing one concrete experimental target. The query you write should feel like something a careful researcher might actually use when trying to find a paragraph, setup description, or protocol evidence that matches the same experiment in the literature. The goal is not to make the wording elegant for its own sake, but to preserve the same experimental intent in a form that is natural, specific, and useful for retrieval.

Before writing anything, first decide whether the record itself supports a meaningful query. Do not judge the fields one by one in isolation. Think about how the fields work together. Ask whether they jointly define a coherent and self-contained experimental target, whether the target remains meaningful once all fields are read together, and whether the resulting request would make sense as a real literature-search problem. Some records look structurally valid at first glance but still should not be kept. This happens when the fields do not jointly define a complete target, when an important dependency is missing, when the target becomes trivial or collapses onto something already fixed, or when the field combination is technically awkward in a way that makes the record unsuitable as a retrieval target.

If the record is usable, write one focused search request that preserves all fixed constraints. You may reorder information and use only strictly equivalent reformulations, such as established abbreviations, standard names or formula variants, exact unit conversions, exact numeric restatements, and light syntactic adjustments. Do not drop any constraint, add any unstated assumption, broaden the target, narrow the target, soften negation, or change the meaning of any number or range. If an exact reformulation is uncertain, keep the original wording.

If the record should not be kept, choose one of the following labels. Use "wrong_constraint" when the record itself does not form a meaningful, coherent, and self-contained retrieval target. Use "low_retrieval_value" when the record can be rewritten faithfully, but the resulting query would still be too weak, too broad, too generic, too common, or too poorly motivated to be useful as a benchmark retrieval query.

Return only one JSON object.

If the record should be kept:
{{"request_query": "<query>", "skip": false, "skip_reason": null}}

If the record should be skipped:
{{"request_query": null, "skip": true, "skip_reason": "<one of the two labels above>"}}

Do not output any explanation, note, or markdown fence.

Field definitions:
{field_definitions_pretty}

Examples:

Example 1
Structured intent:
* cultured_cells.cell_type = primary human hepatocytes
* scaffold_material = collagen I
* synthesis_route = electrospinning

Output:
{{"request_query": "Find experimental setups using primary human hepatocytes cultured in collagen I scaffolds prepared by electrospinning.", "skip": false, "skip_reason": null}}

Example 2
Structured intent:
* applied_potential between -1.2-0.8 V
* electrolytes.chemical = PBS
* measurement_method = CV
* reference_electrode != Ag/AgCl (3 M NaCl)
* working_electrode = GCE

Why skip:
The potential range is not anchored to a clearly fixed reference electrode. In electrochemistry, a voltage range only becomes a stable retrieval condition when its reference system is clearly defined, so this record does not form a reliable retrieval target.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_constraint"}}

Example 3
Structured intent:
* precursors = Al2O3
* sintering.time = 2 h
* synthesis_route != mechanical alloying
* target_material = Al2O3

Why skip:
A query can be written faithfully, but the target collapses onto the precursor and does not define a strong retrieval goal. The resulting request is too weak and poorly motivated to keep as a benchmark query.

Output:
{{"request_query": null, "skip": true, "skip_reason": "low_retrieval_value"}}

Example 4
Structured intent:
* applied_potential = 1 V
* electrolytes.chemical = phosphate buffer
* measurement_method = CV
* working_electrode = GCE

Why skip:
The problem is not just wording. Cyclic voltammetry is a scanned-potential method, so pairing it with a single fixed applied potential creates a field combination that is methodologically mismatched. The record itself is not a good retrieval target in this form.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_constraint"}}

Example 5
Structured intent:
* electrolytes.concentration = 1:1 (v/v)
* measurement_method = cyclic voltammetry
* working_electrode = carbon cloth

Why skip:
The mixture ratio is given without the identities of the components. Because the missing dependency is part of the record itself, this does not define a complete and interpretable retrieval target.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_constraint"}}

Structured record:
{record_pretty_json}
"""

gen_prompt_exploration = """You are given a structured experimental record from {domain} that contains fixed conditions together with one explore field. Your task is to decide whether this record should be kept as a final retrieval query candidate, and if it should, to rewrite it as one concise natural-language research question.

Read the record as describing one concrete experimental target. The fixed fields define the setup that must remain fixed. The explore field is the only part that should remain open. The question you write should feel like something a careful researcher might actually ask when trying to recover one missing experimental detail from the literature under already known conditions.

Before writing anything, first decide whether the record supports a meaningful question at all. Do not judge the fields one by one in isolation. Think about how they work together. Ask whether the fixed conditions jointly define a coherent and self-contained setup, whether the explore field is genuinely appropriate to ask for under those conditions, and whether the resulting question would make sense as a real literature-search problem. Some records look structurally valid but still should not be kept. This happens when the fixed conditions are too weak or incomplete to support the target being asked for, when an important dependency is missing, when the explore target is already strongly implied by the fixed conditions, or when the target remains too broad or too poorly motivated to make a good benchmark question.

If the record is usable, write one focused research question that preserves all fixed constraints and asks directly for the explore field. You may reorder information and use only strictly equivalent reformulations, such as established abbreviations, standard names or formula variants, exact unit conversions, exact numeric restatements, and light syntactic adjustments. Do not drop any fixed condition, add any unstated assumption, broaden the setup, narrow the setup, soften negation, or change the meaning of any number or range. The explore field must stay open and must not be treated as already known.

If the record should not be kept, choose one of the following labels. Use "wrong_constraint" when the fixed conditions themselves do not form a meaningful, coherent, and self-contained retrieval target. Use "wrong_explore_target" when the fixed conditions are broadly interpretable, but the explore field is not an appropriate thing to ask for under those conditions, including cases where it is already strongly implied or where the target lacks enough supporting context to be meaningfully asked. Use "low_retrieval_value" when the record can be rewritten faithfully, but the resulting question would still be too weak, too broad, too generic, too common, or too poorly motivated to be useful as a benchmark retrieval query.

Return only one JSON object.

If the record should be kept:
{{"request_query": "<query>", "skip": false, "skip_reason": null}}

If the record should be skipped:
{{"request_query": null, "skip": true, "skip_reason": "<one of the three labels above>"}}

Do not output any explanation, note, or markdown fence.

Field definitions:
{field_definitions_pretty}

Examples:

Example 1
Structured intent:
* cultured_cells.cell_type = LS174T
* culture_medium.basal_medium != RPMI
* explore field = seeding_density

Output:
{{"request_query": "What seeding density is used for LS174T cells cultured in a medium other than RPMI?", "skip": false, "skip_reason": null}}

Example 2
Structured intent:
* measurement_method != CV
* reference_electrode = Li/Li+
* temperature = room temperature
* working_electrode = stainless steel
* explore field = applied_potential

Why skip:
The fixed conditions remain too broad to support a meaningful question about applied potential. Knowing only that the method is not CV, together with the electrode, reference system, and room temperature, does not define a specific enough experimental context for asking what potential is used.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_explore_target"}}

Example 3
Structured intent:
* measurement_method = LSV
* reference_electrode != Ag/AgCl (KCl, saturated)
* scan_rate = 0.1 mV/s
* working_electrode = stainless steel
* explore field = applied_potential

Why skip:
The question asks for applied potential, but the fixed conditions do not specify the reaction or target clearly enough. The setup remains too under-specified for the requested target, so the explore field is not appropriate under these conditions.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_explore_target"}}

Example 4
Structured intent:
* dopant_or_additive.identity != polyvinyl alcohol (PVA)
* post_treatment.temperature = 1100 °C
* target_material = SrFe12O19
* explore field = post_treatment.type

Why skip:
The fixed conditions already strongly suggest the post-treatment type. Once a high post-treatment temperature is given, asking for the treatment type becomes weak as an open retrieval target because the answer is already heavily implied.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_explore_target"}}

Example 5
Structured intent:
* measurement_method = cyclic voltammetry
* temperature = room temperature
* explore field = working_electrode

Why skip:
A faithful question can be written, but it would still be too broad and weak to serve as a useful benchmark retrieval query.

Output:
{{"request_query": null, "skip": true, "skip_reason": "low_retrieval_value"}}

Example 6
Structured intent:
* electrolytes.chemical = water
* temperature = room temperature
* working_electrode = ITO
* explore field = electrolytes.concentration

Why skip:
The fixed conditions themselves do not form a sound retrieval target. The chemical identity and concentration do not combine into a meaningful electrolyte specification in this form.

Output:
{{"request_query": null, "skip": true, "skip_reason": "wrong_constraint"}}

Structured record:
{record_pretty_json}
"""

eval_prompt = """You are evaluating a rewritten query against the structured experimental record it came from. Your task is to decide whether the query is good enough to keep as a final benchmark retrieval query for {domain}.

Read the structured record as the source of truth. The query should preserve the same target and the same constraints. If the record contains an explore field, the query should keep the fixed conditions intact and directly ask for that field. If there is no explore field, the query should function as a request for matching setup evidence under the stated conditions.

Judge the query on three dimensions.

Faithfulness asks whether the query still means the same thing as the record. A faithful query does not drop conditions, add new ones, soften negation, broaden a concrete value into a looser category, shift a numeric boundary, or otherwise change the retrieval scope.

Clarity asks whether the query is understandable as one focused search request. It does not need to sound elegant, but it should read clearly and not feel confusing, mechanically stitched together, or hard to interpret.

Retrieval usefulness asks whether the query would actually work as a strong final retrieval query for experimental evidence. A query can be faithful and still fail here. This happens when the question is too broad, too weakly constrained, too generic, too unnatural as a real search request, or too poorly motivated. It should also fail here when the surrounding context is not specific enough to support the target being asked for, when the open target is already strongly implied by the fixed conditions, when an important numeric or technical condition is not meaningfully anchored, or when the field combination is mismatched in a way that makes the resulting query scientifically awkward even if it is superficially readable.

After scoring the query, assign one primary issue:
- "wrong_constraint" if the main problem is lack of faithfulness to the record.
- "awkward_wording" if the query is mostly faithful but unclear, awkward, or hard to read.
- "low_retrieval_value" if the query is understandable but still weak as a final retrieval query.
- "none" only if the query is fully good as written.

Use "wrong_constraint" whenever faithfulness = 0. Use "none" only when all three scores are 1.

Then decide whether to keep, revise, or drop the query. If faithfulness = 0, it must be dropped. If faithfulness = 1 and only one of the three scores is positive, it should also be dropped. If faithfulness = 1 and exactly two scores are positive, it should be revised. If all three scores are positive, it should be kept.

Only provide a suggested revision when the problem can be fixed by rewriting the query alone, without changing the underlying structured intent. If the record itself is the main problem, or if the query should be dropped, leave the suggested revision empty.

Keep the reason short and diagnostic. It should point to the main problem rather than restating the whole record.

Return exactly one JSON object in the following format and nothing else:

{{
  "faithfulness": 0 or 1,
  "clarity": 0 or 1,
  "retrieval_usefulness": 0 or 1,
  "primary_issue": "wrong_constraint" | "awkward_wording" | "low_retrieval_value" | "none",
  "final_decision": "keep" | "revise" | "drop",
  "reason": "<short explanation>",
  "suggested_revision": "<revised query or empty string>"
}}

Field definitions:
{field_definitions_pretty}

Structured record:
{record_pretty_json}

Rewritten query:
{generated_query}
"""


FIELD_DEFS = {
    "electrochemistry": {
        "electrolytes.chemical": "The electrolyte-component identity or identities used in the electrochemical system. This may include salts, acids, bases, solvents, ionic liquids, added water, buffer identities, or other explicitly stated electrolyte constituents. Keep the component identity itself rather than its concentration, ratio, pH, or other composition details. Standard naming variants of the same component may be treated as equivalent when they preserve the same chemical identity.",
        "electrolytes.concentration": "The concentration or composition value associated with an electrolyte component, such as 0.1 M, 50 mM, 1 wt%, 10 vol%, 100 ppm, or 1:1 (v/v). Use only values that are clearly tied to the electrolyte composition in the setup. Equivalent unit surface forms may be treated as compatible when they preserve the same concentration meaning. Do not use generic composition descriptions when no specific concentration or composition value is given.",
        "reference_electrode": "The reference-electrode identity used in the setup, such as Ag/AgCl, SCE, RHE, or another explicitly stated reference system. Keep the reference-electrode identity itself rather than surrounding apparatus descriptions or other setup details. Standard naming variants of the same reference electrode may be treated as equivalent when they preserve the same reference-electrode family and identity.",
        "working_electrode": "The working-electrode identity used in the setup, such as Pt, glassy carbon, Bi(111), carbon cloth, or another explicitly stated electrode material or named electrode system. Keep the working-electrode identity itself rather than generic cell descriptions or unrelated apparatus wording. Standard naming variants of the same working electrode may be treated as equivalent when they preserve the same electrode identity.",
        "target_species": "The analyte, reactant, or key electroactive species being studied, detected, oxidized, reduced, or otherwise treated as the main electrochemical target in the setup, such as O2, CO2, H2O2, glucose, nitrate, or another clearly stated target species. Keep the target-species identity itself rather than its concentration, signal description, or surrounding procedural wording. Standard naming variants of the same target species may be treated as equivalent when they preserve the same chemical identity.",
        "temperature": "The experimental temperature condition used in the setup, such as 25 °C, 80 °C, room temperature, or 298 K. Use only values that clearly describe temperature. Equivalent surface forms may be treated as compatible when they preserve the same temperature meaning.",
        "measurement_method": "The electrochemical measurement method or technique used in the setup, such as cyclic voltammetry, chronoamperometry, electrochemical impedance spectroscopy, differential pulse voltammetry, or another clearly stated method. Keep the method identity itself rather than scan rate, potential window, instrument name, or other procedural details. Standard full-name and abbreviation variants of the same method may be treated as equivalent when the method identity is unchanged.",
        "scan_rate": "The scan rate used in voltammetric measurements, such as 50 mV/s, 0.1 V/s, or 10-100 mV/s. Use only values that are clearly stated as scan rate in the setup. Equivalent unit surface forms may be treated as compatible when they preserve the same scan-rate meaning.",
        "applied_potential": "The applied potential, fixed potential, or potential window used in the electrochemical measurement, such as -0.8 V, 0.2 - 1.0 V, or 1.2 V vs Ag/AgCl. Keep the potential condition itself, including reference information when it is part of the stated value. Equivalent surface forms may be treated as compatible when they preserve the same potential meaning."
    },

    "organic_synthesis": {
        "transformation_type": "The reaction class or transformation type carried out in the setup, such as Suzuki coupling, Sonogashira coupling, hydrogenation, oxidation, reduction, cyclization, cycloaddition, amidation, arylation, or borylation. Use the transformation label itself rather than substrate names, product names, catalyst information, or procedural details. Natural wording variation is acceptable, but the underlying transformation meaning should remain unchanged.",
        "substrates": "The key stoichiometric starting materials or reaction partners that are transformed in the setup. Use the substrate or reaction-partner identity itself rather than amounts, equivalents, local paper-specific labels, or procedural wording. Standard naming variants of the same substrate may be treated as equivalent when they preserve the same chemical identity.",
        "catalysts.name": "The catalytic species used in the reaction, such as Pd(PPh3)4, PdCl2(PPh3)2, CuTC, or RuCl3. Use the catalyst identity itself rather than loading, concentration, or surrounding condition details. Standard naming variants of the same catalyst may be treated as equivalent.",
        "catalysts.loading": "The catalyst loading used in the reaction, such as 5 mol %, 10 mol %, 0.3 mmol, or another explicitly stated catalyst amount. Use only values that are clearly tied to the catalyst in the setup. Equivalent surface variants of the same loading expression may be treated as compatible when they preserve the same quantitative meaning.",
        "ligands": "The ligand identity used in the reaction, such as PPh3, BINAP, bpy, or XPhos. Use the ligand identity itself rather than catalyst descriptions, loading, local paper-specific codes, or procedural wording. Standard naming variants of the same ligand may be treated as equivalent.",
        "bases": "The base used in the reaction, such as Cs2CO3, Na2CO3, K3PO4, DBU, or Et3N. Use the base identity itself rather than amount phrases, concentration, or procedural details. Standard naming variants of the same base may be treated as equivalent.",
        "additives": "The functional additive or promoter used in the reaction, such as an oxidant, reductant, acid, salt, or another explicitly added reaction additive. Use the additive identity itself rather than amount phrases, concentration, or procedural details. Standard naming variants of the same additive may be treated as equivalent when they preserve the same chemical identity and role.",
        "solvents": "The solvent or solvent components used as the reaction medium, such as DME, toluene, THF, water, or H2O. Use the solvent identity itself rather than volume, ratio, drying status, or procedural wording. Standard naming variants of the same solvent may be treated as equivalent.",
        "activation_mode": "The high-level reaction-driving or activation mode used in the setup, such as thermal, photochemical, electrochemical, microwave, mechanochemical, sonochemical, or plasma. Use the activation mode itself rather than device details, reactor descriptions, or other surrounding condition wording. Related wording may be matched when the intended activation meaning is clear."
    },

    "cell_culture": {
        "cultured_cells.cell_type": (
            "The cultured cell identity or identities in the setup. "
            "Use the core cell identity needed for stable distinction, such as SH-SY5Y, HepG2, or primary human hepatocytes. "
            "Keep extra descriptors only when they distinguish a biologically different identity, such as primary status, species, stem-cell class, or a specific differentiated form. "
            "Obvious naming variants that clearly refer to the same cell identity may be treated as equivalent, but biologically distinct cell identities should not be merged."
        ),

        "cell_source_type": (
            "The source category of the cultured cells, such as primary, cell line, immortalized, stem-cell-derived, or patient-derived. "
            "Use the most specific supported category when it is clear enough from the setup. "
            "Natural wording variation is acceptable, but the underlying source category should remain unchanged."
        ),

        "culture_medium.basal_medium": (
            "The basal culture medium used in the setup, such as DMEM, RPMI-1640, IMDM, MEM, or Ham's F12. "
            "Keep only the basal medium identity rather than other medium components or procedural details. "
            "Standard naming variants of the same basal medium may be treated as equivalent."
        ),

        "culture_medium.serum.identity": (
            "The serum type used in the culture medium, such as fetal bovine serum, fetal calf serum, horse serum, or human serum. "
            "Keep the serum identity itself rather than its concentration or other preparation details. "
            "Standard naming variants of the same serum type may be treated as equivalent."
        ),

        "culture_medium.serum.concentration": (
            "The serum concentration used in the culture medium, such as 10%, 2%, 0-10%, or serum-free. "
            "Use only values that are clearly tied to serum in the setup. "
            "Serum-free or explicit absence of serum may be treated as 0%."
        ),

        "culture_substrate_or_coating": (
            "The material, matrix, surface, or engineered substrate used for cell attachment, coating, support, or embedding, such as Collagen I, Matrigel, gelatin, laminin, fibronectin, or tissue culture plastic. "
            "Keep the substrate identity itself rather than dimensions, vessel information, or procedural wording. "
            "Obvious naming variants of the same substrate may be treated as equivalent."
        ),

        "seeding_density": (
            "The initial seeding or plating amount used to start the culture. "
            "This is usually expressed as a cell number together with its denominator or culture unit, such as cells/cm^2, cells/mL, or cells/well. "
            "Use only the starting amount, not later density or final concentration values."
        ),

        "culture_mode": (
            "The culture format of the setup, using labels such as 2D, 2.5D, 3D, spheroid, organoid, suspension, chip, Transwell, ALI, co-culture, or monoculture. "
            "Use explicit culture descriptors or very clear setup evidence. "
            "Do not merge different culture modes into a more specific combined label unless that combined form is itself the intended field value."
        ),

        "differentiation_state": (
            "The differentiation status of the cultured cells, such as differentiated, undifferentiated, dedifferentiated, or partially differentiated. "
            "Use this field only when the setup clearly supports that state. "
            "Related biological wording may be matched when the intended differentiation meaning is clear."
        ),

        "confluence": (
            "The confluence value, range, or qualitative confluence state in the setup, such as 80%, 80-85%, near confluence, or full confluence. "
            "Use only values that are clearly expressed as confluence. "
            "Equivalent numeric surface forms may be treated as compatible when they preserve the same confluence meaning."
        )
    },
    
    
    "immunotechnology": {
        "protein_targets.target": "The core protein target intentionally probed in the Western blot, such as Akt, ERK, MARCO, TIM4, SAV1, or MOB1A. Use the target identity itself rather than antibody product descriptions, vendor information, control-role wording, or target-form wording such as phospho or total. Standard naming variants of the same target may be treated as equivalent when they preserve the same protein identity.",
        "protein_targets.target_form": "The probed target form or state in the Western blot, such as phospho, total, cleaved, precursor, mature, or activated. Use this field only when the target form is explicitly stated or clearly indicated by the target wording itself. Related wording may be matched when the intended target-form meaning is clear, but the target identity and the target form should remain distinct.",
        "normalization_control": "The protein used as a loading control or normalization control in the blot, such as GAPDH, beta-actin, ACTB, tubulin, or vinculin. Use the control-protein identity itself rather than surrounding control-role wording, antibody description, or procedural details. Standard naming variants of the same control protein may be treated as equivalent.",
        "sample_type": "The sample category used for blotting, such as cell lysate, whole cell lysate, tissue lysate, total protein extract, nuclear extract, cytoplasmic extract, serum, or plasma. Use the sample category itself rather than cell line names, tissue names, disease labels, treatment groups, lysis-buffer identity, or preparation details. Obvious near-equivalent wording may be treated as compatible when the same sample category is preserved.",
        "lysis_buffer": "The lysis or extraction buffer used to prepare the sample, such as RIPA buffer, NP-40 lysis buffer, urea buffer, Laemmli buffer, or SDS lysis buffer. Use the buffer identity itself rather than full composition details, additives, vendor information, or surrounding preparation procedure. Standard naming variants of the same buffer may be treated as equivalent.",
        "blocking_solution.identity": "The blocking reagent identity or named blocking solution used before antibody incubation, such as BSA, nonfat dry milk, skim milk, casein, normal serum, or Intercept TBS Blocking Buffer. Use the blocking-solution identity itself rather than concentration, dilution, buffer context, incubation conditions, or procedural wording. Standard naming variants of the same blocking solution may be treated as equivalent.",
        "blocking_solution.concentration": "The concentration, dilution, or working-strength expression of the blocking solution, such as 5%, 3%, 1%, 1:10, or 1×. Use only values that are clearly tied to the blocking solution in the setup. Equivalent surface forms of the same concentration meaning may be treated as compatible when they preserve the same quantitative meaning.",
        "gel_percentage": "The SDS-PAGE gel percentage or percentage range used in the blot, such as 8%, 10%, 12%, or 10-12%. Use only the gel percentage information itself rather than gel type wording, electrophoresis context, or procedural details. Equivalent surface variants may be treated as compatible when they preserve the same percentage meaning.",
        "loading_amount": "The sample or protein loading amount per lane or well, such as 20 μg, 30 μg, 50 μg/lane, or 10 μL/well. Use only the loading amount itself rather than sample identity, protein identity, gel context, or procedural wording. Equivalent surface variants of the same loading amount may be treated as compatible when they preserve the same quantitative meaning."
    },

    "inorganic_materials": {
        "target_material": "The target inorganic material system prepared or studied in the setup, such as Al2O3, ZTA, SiC, AlN-Al2O3 composite, or another explicitly stated target material. Use the material identity itself rather than precursor information, additive information, processing details, morphology-only descriptions, or surrounding procedural wording. Standard naming variants of the same material system may be treated as equivalent when they preserve the same material identity.",
        "precursors": "The precursor material or starting powder identity used to prepare the target material, such as Al2O3, Y2O3, AlN, nitrate salts, carbonates, oxides, or another explicitly stated precursor. Use the precursor identity itself rather than amount, purity, supplier information, particle size, or surrounding procedural wording. Standard naming variants of the same precursor may be treated as equivalent when they preserve the same precursor identity.",
        "dopant_or_additive.identity": "The identity of a dopant, additive, sintering aid, or other meaningful composition-changing modifier introduced into the material system, such as Y2O3, MgO, CaF2, or another explicitly stated additive identity. Use the additive identity itself rather than its amount, role wording, or surrounding procedural details. Standard naming variants of the same additive may be treated as equivalent when they preserve the same chemical identity.",
        "dopant_or_additive.amount": "The amount used for a dopant or additive in the material system, such as 1 wt%, 5 mol%, 0.5 at%, 10 vol%, 500 ppm, or another explicitly stated additive amount. Use only values that are clearly tied to the dopant or additive rather than to the target material or another process parameter. Equivalent unit surface forms may be treated as compatible when they preserve the same amount meaning.",
        "synthesis_route": "The high-level synthesis, consolidation, or named material-processing route used in the setup, such as spark plasma sintering, hot pressing, pressureless sintering, solid-state reaction, sol-gel, hydrothermal synthesis, or another clearly stated route. Use the route identity itself rather than atmosphere, temperature, time, pressure, or other processing details. Standard naming variants of the same route may be treated as equivalent when they preserve the same route identity.",
        "atmosphere": "The processing atmosphere used in the relevant synthesis, consolidation, sintering, or post-treatment step, such as air, vacuum, nitrogen, argon, hydrogen, inert gas, reducing atmosphere, or another explicitly stated atmosphere. Use the atmosphere identity itself rather than temperature, pressure, flow rate, or surrounding procedural wording. Standard naming variants of the same atmosphere may be treated as equivalent when they preserve the same atmosphere meaning.",
        "heating_rate": "The heating rate used in thermal processing, such as 5 \u00b0C/min, 10 \u00b0C/min, or 2-10 \u00b0C/min. Use only explicit heating-rate values rather than temperature, time, atmosphere, or other process details. Equivalent unit surface forms may be treated as compatible when they preserve the same heating-rate meaning.",
        "pressing_pressure": "The pressing or consolidation pressure used in shaping, compaction, hot pressing, or another relevant processing step, such as 12 MPa, 20 MPa, or 45 MPa. Use only explicit pressure values rather than load descriptions, temperature, time, or surrounding apparatus wording. Equivalent unit surface forms may be treated as compatible when they preserve the same pressure meaning.",
        "sintering.temperature": "The sintering temperature used for the main sintering step, such as 1200 \u00b0C, 1600 \u00b0C, or 1000-1200 \u00b0C. Use only explicit sintering-temperature values rather than sintering time, atmosphere, pressure, or other process details. Equivalent surface forms may be treated as compatible when they preserve the same temperature meaning.",
        "sintering.time": "The dwell or holding time used for the main sintering step, such as 1 h, 2 h, or 0.5-3 h. Use only explicit sintering-time values rather than sintering temperature, atmosphere, or other process details. Equivalent unit surface forms may be treated as compatible when they preserve the same time meaning.",
        "post_treatment.type": "The type of distinct post-treatment step performed before or after the main sintering step, such as annealing, calcination, oxidation, reduction, quenching, or heat treatment. Use the treatment identity itself rather than its temperature, time, atmosphere, purpose wording, or other procedural details. Standard naming variants of the same treatment type may be treated as equivalent when they preserve the same treatment identity.",
        "post_treatment.temperature": "The temperature used for a distinct post-treatment step, such as 600 \u00b0C, 900 \u00b0C, or 500-800 \u00b0C. Use only explicit post-treatment temperature values rather than treatment type, time, atmosphere, or other process details. Equivalent surface forms may be treated as compatible when they preserve the same temperature meaning.",
        "post_treatment.time": "The time used for a distinct post-treatment step, such as 1 h, 4 h, or 0.5-2 h. Use only explicit post-treatment time values rather than treatment type, temperature, or other process details. Equivalent unit surface forms may be treated as compatible when they preserve the same time meaning."
    },

    "polymer_materials": {
        "polymer_system": "The main photocurable polymer, resin, monomer, oligomer, or crosslinkable network system used in the curing setup, such as GelMA, PEGDA, HDDA, UDMA, or another explicitly named main polymer system. Use the polymer-system identity itself rather than its concentration, additives, or surrounding formulation details. Standard naming variants of the same polymer system may be treated as equivalent.",
        "polymer_concentration": "The concentration, fraction, loading, or amount-per-volume value used for the main polymer system in the curing setup, such as 10 wt%, 15% w/v, 20 wt%, 50 mg/mL, or 1 M. Use only values that are clearly tied to the main polymer system rather than to photoinitiators, crosslinking agents, solvents, or other additives. Straightforward unit-compatible numeric forms may be treated as equivalent when they preserve the same concentration meaning.",
        "photoinitiators.name": "The identity of the photoinitiator used in the curing setup, such as LAP, Irgacure 2959, TPO, BAPO, eosin Y, or camphorquinone. Use the initiator identity itself rather than its loading, supplier information, or other formulation details. Standard naming variants of the same photoinitiator may be treated as equivalent.",
        "photoinitiators.loading": "The loading or concentration used for the photoinitiator in the curing setup, such as 0.5 wt%, 1 mol%, 0.1% w/v, 5 mg/mL, or 2-10 wt%. Use only values that are clearly tied to a photoinitiator rather than to the main polymer system or another additive. Straightforward unit-compatible numeric forms may be treated as equivalent when they preserve the same loading meaning.",
        "crosslinking_agent.name": "The identity of an explicitly added crosslinking agent or other distinct reactive formulation additive used together with the main polymer system, such as PEGDA, TEA, NVP, or another clearly named auxiliary reactive component. Use the component identity itself rather than its loading or surrounding formulation details. Standard naming variants of the same component may be treated as equivalent.",
        "crosslinking_agent.loading": "The loading or concentration used for an explicitly added crosslinking agent or other retained reactive crosslinking-related additive in the curing setup, such as 0.5 wt%, 1 mol%, 0.1% w/v, 5 mg/mL, or 2-10 wt%. Use only values that are clearly tied to a crosslinking agent rather than to the main polymer system, a photoinitiator, or another additive. Straightforward unit-compatible numeric forms may be treated as equivalent when they preserve the same loading meaning.",
        "wavelength": "The irradiation wavelength used in the curing setup, such as 365 nm, 405 nm, or 395-400 nm. Use only explicit wavelength values rather than color descriptions or generic source descriptions. Equivalent numeric surface forms may be treated as compatible when they preserve the same wavelength meaning.",
        "intensity": "The light intensity or irradiance used in the curing setup, such as 5 mW/cm^2, 10 mW/cm^2, or 0.1 W/cm^2. Use only explicit intensity values rather than wavelength, exposure time, or source descriptions. Straightforward unit-compatible numeric forms may be treated as equivalent when they preserve the same intensity meaning.",
        "exposure_time": "The irradiation or exposure time used in the curing setup, such as 10 s, 30 s, 2 min, or 0.5-1 min. Use only explicit exposure-time values rather than wavelength, intensity, or source descriptions. Straightforward unit-compatible numeric forms may be treated as equivalent when they preserve the same exposure-time meaning.",
        "light_source_type": "The type of light source used for curing, such as LED, UV LED, laser, projector, mercury lamp, xenon lamp, halogen lamp, or UV lamp. Use the source identity itself rather than wavelength, intensity, exposure time, model number, or supplier information. Standard naming variants of the same source type may be treated as equivalent."
    }
}


# from collections import Counter
# import json

# counter = Counter()
# with open("seed_data/electrochemistry/constraints/sampled_constraints_stage1.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         obj = json.loads(line)
#         counter[obj["candidate_id"]] += 1

# dups = {k: v for k, v in counter.items() if v > 1}
# print(len(dups))
# print(list(dups.items())[:20])

# python scripts/gen_query/prompt.py

