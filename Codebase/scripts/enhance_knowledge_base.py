"""
Script to add detailed symptoms to diseases in medical_knowledge.txt
"""

with open('data/medical_knowledge.txt', 'r') as f:
    content = f.read()

docs = content.split('\n---\n')

# Enhancements: key = disease name prefix, value = enhanced version with symptoms
enhancements = {
    'CAP': 'CAP: Community-acquired pneumonia. Streptococcus pneumoniae common. Symptoms: fever, productive cough, pleuritic chest pain, dyspnea. CURB-65 score. Treatment: beta-lactam plus macrolide.',
    'STEMI vs NSTEMI': 'STEMI vs NSTEMI: ST elevation vs non-elevation MI. Symptoms: severe chest pain, diaphoresis, nausea, dyspnea. Emergency PCI for STEMI.',
    'HFrEF vs HFpEF': 'HFrEF vs HFpEF: Heart failure subtypes. Symptoms: dyspnea, orthopnea, edema, fatigue. HFrEF: reduced ejection fraction. HFpEF: preserved ejection fraction.',
    'AKI': 'AKI: Acute kidney injury. Symptoms: oliguria, edema, confusion, nausea. Causes: pre-renal, intrinsic, post-renal. Elevated creatinine. Treatment: treat cause, fluids, avoid nephrotoxins.',
    'IBS': 'IBS: Irritable bowel syndrome. Symptoms: abdominal pain, bloating, diarrhea or constipation, mucus in stool. Treatment: low FODMAP diet, fiber, antispasmodics.',
    'Bipolar Disorder': 'Bipolar Disorder: Mood disorder. Symptoms: Manic episodes with elevated mood, decreased sleep, impulsivity. Depressive episodes with low mood, anhedonia. Treatment: mood stabilizers, antipsychotics.',
    'Epilepsy': 'Epilepsy: Recurrent seizures. Symptoms: convulsions, loss of consciousness, aura, post-ictal confusion. Focal vs generalized. EEG, MRI. Treatment: antiepileptic drugs.',
    'Iron Deficiency Anemia': 'Iron Deficiency Anemia: Microcytic anemia. Symptoms: fatigue, weakness, pallor, pica, restless legs, brittle nails. Low ferritin. Treatment: oral iron supplementation.',
    'B12 Deficiency': 'B12 Deficiency: Macrocytic anemia. Symptoms: fatigue, glossitis, paresthesias, ataxia, memory loss. Pernicious anemia. Treatment: B12 supplementation.',
    'Folate Deficiency': 'Folate Deficiency: Macrocytic anemia. Symptoms: fatigue, pallor, glossitis, diarrhea. No neurologic symptoms. Treatment: folic acid.',
    'Hemolytic Anemia': 'Hemolytic Anemia: Premature RBC destruction. Symptoms: jaundice, dark urine, fatigue, splenomegaly. Elevated LDH, indirect bilirubin. Treatment: treat cause.',
    'Sickle Cell Disease': 'Sickle Cell Disease: HbSS. Symptoms: pain crises, dactylitis, fatigue, jaundice, priapism. Vaso-occlusive crises. Treatment: hydroxyurea, transfusions.',
    'Thalassemia': 'Thalassemia: Hemoglobin synthesis disorder. Symptoms: fatigue, pallor, jaundice, splenomegaly. Microcytic anemia. Treatment: transfusions, iron chelation.',
    'ALL': 'ALL: Acute lymphoblastic leukemia. Symptoms: fever, fatigue, bruising, bone pain, lymphadenopathy. Childhood cancer. Treatment: chemotherapy, CNS prophylaxis.',
    'CLL': 'CLL: Chronic lymphocytic leukemia. Symptoms: often asymptomatic, lymphadenopathy, fatigue, infections. Lymphocytosis, smudge cells. Treatment: watch and wait.',
    'AML': 'AML: Acute myeloid leukemia. Symptoms: fever, fatigue, bleeding, gum hypertrophy. Blasts, Auer rods. Treatment: chemotherapy, stem cell transplant.',
    'CML': 'CML: Chronic myeloid leukemia. Symptoms: often asymptomatic, fatigue, splenomegaly. Philadelphia chromosome BCR-ABL. Treatment: tyrosine kinase inhibitors.',
    'Hodgkin Lymphoma': 'Hodgkin Lymphoma: Lymphoid malignancy. Symptoms: painless lymphadenopathy, B symptoms, pruritus. Reed-Sternberg cells. Treatment: ABVD chemotherapy.',
    'Non-Hodgkin Lymphoma': 'Non-Hodgkin Lymphoma: Lymphoid malignancy. Symptoms: lymphadenopathy, B symptoms, extranodal involvement. Treatment: R-CHOP, rituximab-based regimens.',
    'Multiple Myeloma': 'Multiple Myeloma: Plasma cell malignancy. Symptoms: bone pain, fatigue, infections, hypercalcemia, renal failure. CRAB criteria. Treatment: proteasome inhibitors.',
    'DVT': 'DVT: Deep vein thrombosis. Symptoms: unilateral leg pain, swelling, warmth, erythema. Wells score, D-dimer, ultrasound. Treatment: anticoagulation.',
    'Pulmonary Embolism': 'Pulmonary Embolism: Clot in pulmonary artery. Symptoms: sudden dyspnea, pleuritic chest pain, tachycardia, hemoptysis. CTPA. Treatment: anticoagulation.',
    'Cirrhosis': 'Cirrhosis: End-stage liver disease. Symptoms: ascites, varices, encephalopathy, jaundice, spider angiomas. Causes: alcohol, hepatitis. Treatment: transplant.',
    'Hepatitis B': 'Hepatitis B: DNA virus. Symptoms: jaundice, fatigue, nausea, RUQ pain, dark urine. HBsAg positive. Treatment: entecavir, tenofovir.',
    'Hepatitis C': 'Hepatitis C: RNA virus. Symptoms: often asymptomatic, fatigue, jaundice. Anti-HCV. Treatment: direct-acting antivirals.',
    'NAFLD': 'NAFLD: Non-alcoholic fatty liver. Symptoms: often asymptomatic, RUQ discomfort, fatigue. Obesity, diabetes risk. Treatment: weight loss, exercise.',
    'Acute Pancreatitis': 'Acute Pancreatitis: Pancreas inflammation. Symptoms: severe epigastric pain radiating to back, nausea, vomiting. Elevated lipase. Treatment: NPO, IV fluids.',
    'Chronic Pancreatitis': 'Chronic Pancreatitis: Chronic pancreas inflammation. Symptoms: chronic pain, steatorrhea, weight loss, diabetes. Calcifications. Treatment: enzyme replacement.',
    'Diverticulitis': 'Diverticulitis: Colonic diverticula inflammation. Symptoms: LLQ pain, fever, nausea, constipation. CT abdomen. Treatment: antibiotics.',
    'Appendicitis': 'Appendicitis: Appendix inflammation. Symptoms: RLQ pain, McBurney point tenderness, migratory pain, fever, nausea. Treatment: appendectomy.',
    'Acute Cholecystitis': 'Acute Cholecystitis: Gallbladder inflammation. Symptoms: RUQ pain, Murphy sign, fever, nausea. Ultrasound. Treatment: cholecystectomy.',
    'Choledocholithiasis': 'Choledocholithiasis: CBD stones. Symptoms: jaundice, RUQ pain, fever (Charcot triad), dark urine. Elevated bilirubin. Treatment: ERCP.',
    'UTI - Cystitis': 'UTI - Cystitis: Bladder infection. Symptoms: dysuria, frequency, urgency, suprapubic pain, hematuria. E. coli common. Treatment: nitrofurantoin, TMP-SMX.',
    'Pyelonephritis': 'Pyelonephritis: Kidney infection. Symptoms: fever, flank pain, CVA tenderness, nausea, dysuria. Treatment: fluoroquinolone or ceftriaxone.',
    'BPH': 'BPH: Benign prostatic hyperplasia. Symptoms: weak stream, hesitancy, frequency, nocturia, incomplete emptying. Treatment: alpha-blockers, 5-ARI.',
    'Prostate Cancer': 'Prostate Cancer: Adenocarcinoma. Symptoms: often asymptomatic, LUTS, hematuria, bone pain if metastatic. PSA screening. Treatment: surveillance, prostatectomy.',
    'Testicular Cancer': 'Testicular Cancer: Germ cell tumor. Symptoms: painless testicular mass, heavy sensation, gynecomastia. Young men 15-35. Treatment: orchiectomy.',
    'Erectile Dysfunction': 'Erectile Dysfunction: Inability to maintain erection. Symptoms: difficulty with erections, reduced libido. Causes: vascular, neurogenic. Treatment: PDE-5 inhibitors.',
    'Urolithiasis': 'Urolithiasis: Kidney stones. Symptoms: severe colicky flank pain radiating to groin, hematuria, nausea. Non-contrast CT. Treatment: hydration, lithotripsy.',
    'Poststreptococcal GN': 'Poststreptococcal GN: Immune complex GN. Symptoms: hematuria, edema, hypertension, oliguria, periorbital swelling. Low C3. Treatment: supportive.',
    'IgA Nephropathy': 'IgA Nephropathy: Mesangial IgA deposition. Symptoms: episodic gross hematuria after URI, hypertension. Treatment: ACE inhibitors.',
    'Nephrotic Syndrome': 'Nephrotic Syndrome: Glomerular disease. Symptoms: edema, foamy urine, fatigue. Proteinuria, hypoalbuminemia. Causes: minimal change, FSGS.',
    'Nephritic Syndrome': 'Nephritic Syndrome: Glomerular disease. Symptoms: hematuria, oliguria, hypertension, periorbital edema. RBC casts. Post-infectious GN.',
    'ADPKD': 'ADPKD: Autosomal dominant PKD. Symptoms: hypertension, flank pain, hematuria, UTIs. Multiple bilateral cysts. Complications: ESRD.',
    'Tuberculosis': 'Tuberculosis: Mycobacterium tuberculosis. Symptoms: chronic cough, hemoptysis, night sweats, weight loss, fever. AFB smear. Treatment: RIPE therapy.',
    'HIV/AIDS': 'HIV/AIDS: Human immunodeficiency virus. Symptoms: fever, rash, lymphadenopathy. AIDS with opportunistic infections. CD4 <200. Treatment: antiretroviral therapy.',
    'PCP': 'PCP: Pneumocystis pneumonia. Symptoms: dyspnea, dry cough, fever, hypoxia. CD4 <200. Elevated LDH. Treatment: TMP-SMX, steroids.',
    'Sepsis': 'Sepsis: Life-threatening organ dysfunction. Symptoms: fever, tachycardia, tachypnea, altered mental status, hypotension. SOFA score. Treatment: fluids, antibiotics.',
    'Septic Shock': 'Septic Shock: Sepsis with hypotension. Symptoms: hypotension requiring vasopressors, organ dysfunction, elevated lactate. Treatment: norepinephrine.',
    'Infective Endocarditis': 'Infective Endocarditis: Heart valve infection. Symptoms: fever, new murmur, splinter hemorrhages, Osler nodes. Duke criteria. Treatment: prolonged IV antibiotics.',
    'Bacterial Meningitis': 'Bacterial Meningitis: Meninges infection. Symptoms: fever, headache, nuchal rigidity, photophobia, altered mental status. S. pneumoniae. Treatment: ceftriaxone, vancomycin.',
    'Viral Meningitis': 'Viral Meningitis: Meninges infection. Symptoms: fever, headache, neck stiffness, photophobia. Enteroviruses. Treatment: supportive.',
    'Encephalitis': 'Encephalitis: Brain inflammation. Symptoms: altered mental status, seizures, fever, headache, focal deficits. HSV common. Treatment: acyclovir.',
    'Cellulitis': 'Cellulitis: Skin infection. Symptoms: erythema, warmth, tenderness, swelling, fever. Streptococcus, Staphylococcus. Treatment: cephalexin.',
    'Osteomyelitis': 'Osteomyelitis: Bone infection. Symptoms: bone pain, fever, swelling, erythema over affected area. S. aureus. MRI. Treatment: prolonged antibiotics.',
    'Lyme Disease': 'Lyme Disease: Tick-borne illness. Symptoms: erythema migrans, fever, headache, fatigue, facial palsy, arthritis. Borrelia burgdorferi. Treatment: doxycycline.',
    'Malaria': 'Malaria: Plasmodium infection. Symptoms: cyclic fever, rigors, sweats, headache, myalgias, anemia, splenomegaly. Blood smear. Treatment: artemisinin-based therapy.',
    'Osteoporosis': 'Osteoporosis: Decreased bone density. Symptoms: often asymptomatic until fracture, back pain, height loss, kyphosis. T-score ≤-2.5. Treatment: bisphosphonates.',
    'Vitamin D Deficiency': 'Vitamin D Deficiency: Low vitamin D. Symptoms: bone pain, muscle weakness, fatigue, depression. Treatment: vitamin D supplementation.',
    'Hypercalcemia': 'Hypercalcemia: Elevated calcium. Symptoms: stones, bones, groans, psychiatric overtones. Kidney stones, bone pain, constipation, confusion. Treatment: IV fluids.',
    'Hypocalcemia': 'Hypocalcemia: Low calcium. Symptoms: tetany, Chvostek sign, Trousseau sign, paresthesias, seizures. Hypoparathyroidism. Treatment: calcium supplementation.',
    'Hyperkalemia': 'Hyperkalemia: Elevated potassium. Symptoms: muscle weakness, paresthesias, palpitations. Peaked T waves, widened QRS. Treatment: calcium gluconate, insulin, dialysis.',
    'Hypokalemia': 'Hypokalemia: Low potassium. Symptoms: muscle weakness, cramps, constipation, palpitations. U waves. GI losses, diuretics. Treatment: oral or IV potassium.',
    'Hyponatremia': 'Hyponatremia: Low sodium. Symptoms: headache, confusion, seizures, nausea, lethargy. SIADH, CHF, cirrhosis. Treatment: fluid restriction, hypertonic saline.',
    'Hypernatremia': 'Hypernatremia: Elevated sodium. Symptoms: thirst, confusion, lethargy, seizures, irritability. Water loss, diabetes insipidus. Treatment: free water replacement.',
    'Metabolic Acidosis': 'Metabolic Acidosis: Low pH, low HCO3. Symptoms: Kussmaul breathing, confusion, fatigue. Anion gap vs normal gap. Treatment: treat cause.',
    'Metabolic Alkalosis': 'Metabolic Alkalosis: High pH, high HCO3. Symptoms: muscle cramps, tetany, arrhythmias. Vomiting, diuretics. Treatment: saline if volume depleted.',
    'Respiratory Acidosis': 'Respiratory Acidosis: Low pH, high PaCO2. Symptoms: confusion, drowsiness, headache. Hypoventilation, COPD. Treatment: improve ventilation.',
    'Respiratory Alkalosis': 'Respiratory Alkalosis: High pH, low PaCO2. Symptoms: lightheadedness, paresthesias, tetany. Hyperventilation, anxiety. Treatment: treat cause.',
    'DKA': 'DKA: Diabetic ketoacidosis. Symptoms: Kussmaul breathing, fruity breath, nausea, vomiting, abdominal pain, confusion. Glucose >250, pH <7.3. Treatment: IV fluids, insulin.',
    'HHS': 'HHS: Hyperosmolar hyperglycemic state. Symptoms: altered mental status, severe dehydration, confusion, seizures. Glucose >600. Treatment: aggressive fluids, insulin.',
    'Hypoglycemia': 'Hypoglycemia: Low blood sugar. Symptoms: tremor, diaphoresis, palpitations, confusion, seizures, loss of consciousness. Treatment: oral glucose, IV dextrose, glucagon.'
}

# Apply enhancements
count = 0
for key, enhanced in enhancements.items():
    for i, doc in enumerate(docs):
        if doc.strip().startswith(key + ':'):
            docs[i] = enhanced
            count += 1
            break

# Write back
with open('data/medical_knowledge.txt', 'w') as f:
    f.write('\n---\n'.join(docs))

print(f'✅ Enhanced {count} disease entries with detailed symptoms')
print(f'✅ Total documents in knowledge base: {len(docs)}')
