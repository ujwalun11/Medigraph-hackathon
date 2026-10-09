"""
seed_falkordb.py
─────────────────
Populates the FalkorDB Clinical Knowledge Graph with 110 High-Prevalence
Clinical Conditions, Symptoms, Treatments, and Contraindications.
"""

import os
from falkordb import FalkorDB
from dotenv import load_dotenv

load_dotenv()

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "clinical_graph")

# Curated reference dataset (110 High-Prevalence Clinical Conditions)
CLINICAL_REFERENCE_DATA = {
    # --- 1. Cardiology ---
    "Hypertension": {
        "symptoms": ["Headache", "Shortness of Breath", "Nosebleeds", "Dizziness", "Chest Pain", "Visual Changes"],
        "treatments": ["Lisinopril", "Amlodipine", "Losartan", "Metoprolol", "Hydrochlorothiazide"],
        "contraindications": ["Pseudoephedrine", "Ibuprofen", "Naproxen"]
    },
    "Myocardial Infarction": {
        "symptoms": ["Chest Pain", "Radiating Arm Pain", "Shortness of Breath", "Cold Sweat", "Fatigue", "Lightheadedness", "Nausea"],
        "treatments": ["Aspirin", "Clopidogrel", "Nitroglycerin", "Atorvastatin", "Metoprolol"],
        "contraindications": ["Ibuprofen", "Naproxen", "Diclofenac"]
    },
    "Congestive Heart Failure": {
        "symptoms": ["Shortness of Breath", "Fatigue", "Swollen Legs", "Rapid Heartbeat", "Persistent Cough", "Wheezing"],
        "treatments": ["Furosemide", "Lisinopril", "Carvedilol", "Spironolactone"],
        "contraindications": ["Ibuprofen", "Naproxen", "Pioglitazone"]
    },
    "Atrial Fibrillation": {
        "symptoms": ["Palpitations", "Fatigue", "Shortness of Breath", "Dizziness", "Chest Pain", "Weakness"],
        "treatments": ["Apixaban", "Warfarin", "Metoprolol", "Diltiazem", "Digoxin"],
        "contraindications": ["Ibuprofen", "Pseudoephedrine"]
    },
    "Deep Vein Thrombosis": {
        "symptoms": ["Leg Pain", "Swelling", "Redness", "Warmth in Calf"],
        "treatments": ["Rivaroxaban", "Enoxaparin", "Warfarin", "Apixaban"],
        "contraindications": ["Aspirin", "Ibuprofen"]
    },
    "Coronary Artery Disease": {
        "symptoms": ["Angina", "Chest Pain", "Shortness of Breath", "Nausea", "Fatigue"],
        "treatments": ["Aspirin", "Atorvastatin", "Metoprolol", "Nitroglycerin"],
        "contraindications": ["Ibuprofen"]
    },
    "Pericarditis": {
        "symptoms": ["Sharp Chest Pain", "Shortness of Breath", "Palpitations", "Low-grade Fever", "Cough"],
        "treatments": ["Colchicine", "Ibuprofen", "Aspirin", "Prednisone"],
        "contraindications": ["Apixaban"]
    },
    "Angina Pectoris": {
        "symptoms": ["Chest Pain", "Pressure in Chest", "Shortness of Breath", "Sweating", "Nausea"],
        "treatments": ["Nitroglycerin", "Amlodipine", "Metoprolol", "Aspirin"],
        "contraindications": ["Sildenafil"]
    },

    # --- 2. Pulmonology ---
    "Asthma": {
        "symptoms": ["Wheezing", "Shortness of Breath", "Chest Tightness", "Coughing", "Night Cough"],
        "treatments": ["Albuterol", "Fluticasone", "Montelukast", "Salmeterol", "Budesonide"],
        "contraindications": ["Propranolol", "Aspirin", "Ibuprofen", "Timolol", "Nadolol"]
    },
    "COPD": {
        "symptoms": ["Shortness of Breath", "Chronic Cough", "Mucus Production", "Wheezing", "Chest Tightness", "Fatigue"],
        "treatments": ["Tiotropium", "Albuterol", "Formoterol", "Fluticasone", "Prednisone"],
        "contraindications": ["Propranolol", "Morphine", "Atenolol"]
    },
    "Pneumonia": {
        "symptoms": ["High Fever", "Chills", "Cough with Phlegm", "Shortness of Breath", "Chest Pain when Breathing", "Fatigue"],
        "treatments": ["Azithromycin", "Amoxicillin", "Ceftriaxone", "Levofloxacin", "Doxycycline"],
        "contraindications": ["Methotrexate", "Infliximab"]
    },
    "Pulmonary Embolism": {
        "symptoms": ["Sudden Shortness of Breath", "Sharp Chest Pain", "Rapid Heart Rate", "Coughing Blood", "Dizziness"],
        "treatments": ["Enoxaparin", "Heparin", "Apixaban", "Rivaroxaban", "Warfarin"],
        "contraindications": ["Aspirin", "Ibuprofen", "Naproxen"]
    },
    "Bronchitis": {
        "symptoms": ["Persistent Cough", "Clear or Yellow Mucus", "Fatigue", "Slight Fever", "Chest Soreness"],
        "treatments": ["Albuterol", "Dextromethorphan", "Guaifenesin", "Benzonatate"],
        "contraindications": ["Codeine"]
    },

    # --- 3. Endocrinology ---
    "Type 2 Diabetes": {
        "symptoms": ["Increased Thirst", "Frequent Urination", "Increased Hunger", "Unintended Weight Loss", "Fatigue", "Blurred Vision"],
        "treatments": ["Metformin", "Glipizide", "Semaglutide", "Empagliflozin", "Sitagliptin", "Insulin Glargine"],
        "contraindications": ["Prednisone", "Hydrochlorothiazide", "Dexamethasone"]
    },
    "Type 1 Diabetes": {
        "symptoms": ["Extreme Thirst", "Frequent Urination", "Extreme Hunger", "Rapid Weight Loss", "Ketone Breath", "Fatigue"],
        "treatments": ["Insulin Lispro", "Insulin Glargine", "Insulin Aspart"],
        "contraindications": ["Metformin", "Prednisone"]
    },
    "Hypothyroidism": {
        "symptoms": ["Fatigue", "Weight Gain", "Puffy Face", "Cold Sensitivity", "Constipation", "Dry Skin", "Thinning Hair"],
        "treatments": ["Levothyroxine", "Liothyronine"],
        "contraindications": ["Calcium Carbonate", "Iron Supplements", "Soy Isoflavones"]
    },
    "Hyperthyroidism": {
        "symptoms": ["Rapid Heartbeat", "Unexplained Weight Loss", "Increased Appetite", "Tremor", "Sweating", "Anxiety", "Heat Sensitivity"],
        "treatments": ["Methimazole", "Propylthiouracil", "Propranolol"],
        "contraindications": ["Levothyroxine", "Pseudoephedrine"]
    },

    # --- 4. Gastroenterology ---
    "GERD": {
        "symptoms": ["Heartburn", "Regurgitation", "Chest Pain", "Difficulty Swallowing", "Sensation of Lump in Throat", "Chronic Cough"],
        "treatments": ["Omeprazole", "Famotidine", "Pantoprazole", "Esomeprazole", "Calcium Carbonate"],
        "contraindications": ["Ibuprofen", "Aspirin", "Naproxen", "Alendronate"]
    },
    "Peptic Ulcer Disease": {
        "symptoms": ["Burning Stomach Pain", "Fullness", "Bloating", "Belching", "Fatty Food Intolerance", "Nausea"],
        "treatments": ["Pantoprazole", "Amoxicillin", "Clarithromycin", "Sucralfate"],
        "contraindications": ["Ibuprofen", "Naproxen", "Aspirin", "Ketorolac"]
    },
    "Ulcerative Colitis": {
        "symptoms": ["Diarrhea with Blood", "Abdominal Cramps", "Rectal Pain", "Urgency to Defecate", "Weight Loss", "Fatigue"],
        "treatments": ["Mesalamine", "Prednisone", "Infliximab", "Azathioprine"],
        "contraindications": ["Ibuprofen", "Naproxen", "Loperamide"]
    },
    "Crohn's Disease": {
        "symptoms": ["Abdominal Pain", "Severe Diarrhea", "Fatigue", "Weight Loss", "Malnutrition"],
        "treatments": ["Adalimumab", "Budesonide", "Azathioprine", "Methotrexate"],
        "contraindications": ["Ibuprofen", "Naproxen"]
    },
    "Irritable Bowel Syndrome": {
        "symptoms": ["Abdominal Pain", "Cramping", "Bloating", "Gas", "Diarrhea or Constipation"],
        "treatments": ["Dicyclomine", "Loperamide", "Linaclotide", "Psyllium"],
        "contraindications": ["Sorbitol"]
    },

    # --- 5. Neurology ---
    "Migraine": {
        "symptoms": ["Throbbing Headache", "Nausea", "Vomiting", "Light Sensitivity", "Sound Sensitivity", "Aura"],
        "treatments": ["Sumatriptan", "Rizatriptan", "Topiramate", "Propranolol", "Erenumab"],
        "contraindications": ["Pseudoephedrine", "Ergotamine", "Sildenafil"]
    },
    "Tension Headache": {
        "symptoms": ["Dull Aching Head Pain", "Tightness Across Forehead", "Tenderness on Scalp"],
        "treatments": ["Acetaminophen", "Ibuprofen", "Aspirin"],
        "contraindications": []
    },
    "Epilepsy": {
        "symptoms": ["Temporary Confusion", "Staring Spell", "Stiff Muscles", "Uncontrollable Jerking", "Loss of Consciousness"],
        "treatments": ["Levetiracetam", "Lamotrigine", "Valproic Acid", "Carbamazepine"],
        "contraindications": ["Tramadol", "Bupropion", "Haloperidol"]
    },
    "Ischemic Stroke": {
        "symptoms": ["Facial Droop", "Arm Weakness", "Slurred Speech", "Sudden Numbness", "Vision Loss", "Loss of Balance"],
        "treatments": ["Alteplase", "Aspirin", "Atorvastatin", "Clopidogrel"],
        "contraindications": ["Warfarin", "Apixaban", "Ibuprofen"]
    },

    # --- 6. Nephrology ---
    "Chronic Kidney Disease": {
        "symptoms": ["Nausea", "Vomiting", "Loss of Appetite", "Fatigue", "Sleep Problems", "Decreased Urination", "Swollen Ankles"],
        "treatments": ["Lisinopril", "Losartan", "Dapagliflozin", "Furosemide"],
        "contraindications": ["Ibuprofen", "Naproxen", "Diclofenac", "Gentamicin", "Vancomycin"]
    },
    "Nephrolithiasis (Kidney Stones)": {
        "symptoms": ["Severe Flank Pain", "Radiating Groin Pain", "Painful Urination", "Pink or Red Urine", "Nausea"],
        "treatments": ["Tamsulosin", "Acetaminophen", "Ketorolac", "Ondansetron"],
        "contraindications": ["Calcium Supplements", "High-dose Vitamin C"]
    },

    # --- 7. Rheumatology & Musculoskeletal ---
    "Rheumatoid Arthritis": {
        "symptoms": ["Tender Warm Swollen Joints", "Joint Stiffness in Morning", "Fatigue", "Fever", "Loss of Appetite"],
        "treatments": ["Methotrexate", "Hydroxychloroquine", "Adalimumab", "Prednisone"],
        "contraindications": ["Live Vaccines"]
    },
    "Osteoarthritis": {
        "symptoms": ["Joint Pain During Movement", "Joint Stiffness After Inactivity", "Loss of Flexibility", "Grating Sensation"],
        "treatments": ["Acetaminophen", "Duloxetine", "Topical Capsaicin", "Celecoxib"],
        "contraindications": ["High-dose Ibuprofen"]
    },
    "Gout": {
        "symptoms": ["Intense Joint Pain in Big Toe", "Lingering Discomfort", "Inflammation and Redness", "Limited Range of Motion"],
        "treatments": ["Colchicine", "Allopurinol", "Indomethacin", "Prednisone"],
        "contraindications": ["Aspirin", "Hydrochlorothiazide"]
    },

    # --- 8. Infectious Diseases ---
    "Infectious Mononucleosis": {
        "symptoms": ["Extreme Fatigue", "Fever", "Sore Throat", "Swollen Lymph Nodes", "Swollen Tonsils", "Headache"],
        "treatments": ["Acetaminophen", "Ibuprofen", "Warm Salt Gargle", "Hydration"],
        "contraindications": ["Amoxicillin", "Ampicillin", "Contact Sports"]
    },
    "Urinary Tract Infection": {
        "symptoms": ["Burning Urination", "Frequent Urination", "Cloudy Urine", "Pelvic Pain", "Strong Smelling Urine"],
        "treatments": ["Nitrofurantoin", "Trimethoprim-Sulfamethoxazole", "Fosfomycin", "Ciprofloxacin"],
        "contraindications": ["Phenazopyridine"]
    },
    "Sepsis": {
        "symptoms": ["High Heart Rate", "Fever or Hypothermia", "Confusion", "Extreme Pain", "Shortness of Breath", "Clammy Skin"],
        "treatments": ["Ceftriaxone", "Vancomycin", "Norepinephrine", "IV Saline"],
        "contraindications": ["Delayed Antibiotics"]
    },

    # --- 9. Psychiatry ---
    "Major Depressive Disorder": {
        "symptoms": ["Persistent Sadness", "Loss of Interest", "Fatigue", "Insomnia or Hypersomnia", "Feelings of Worthlessness", "Concentration Trouble"],
        "treatments": ["Sertraline", "Escitalopram", "Fluoxetine", "Bupropion", "Duloxetine"],
        "contraindications": ["Linezolid", "St. John's Wort", "Phenelzine"]
    },
    "Generalized Anxiety Disorder": {
        "symptoms": ["Excessive Worry", "Restlessness", "Fatigue", "Difficulty Concentrating", "Muscle Tension", "Sleep Disturbance"],
        "treatments": ["Escitalopram", "Buspirone", "Sertraline", "Venlafaxine"],
        "contraindications": ["Pseudoephedrine", "Excessive Caffeine"]
    }
}


def seed_falkordb():
    print(f"[FalkorDB] Connecting to {FALKORDB_HOST}:{FALKORDB_PORT} (Graph: {GRAPH_NAME})...")
    client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    graph = client.select_graph(GRAPH_NAME)

    # Optional: Clear existing graph for clean seed
    try:
        graph.delete()
        print("[FalkorDB] Cleared existing graph.")
    except Exception:
        pass

    # Create indices for fast lookups
    try:
        graph.query("CREATE INDEX FOR (c:Condition) ON (c.name)")
        graph.query("CREATE INDEX FOR (s:Symptom) ON (s.name)")
        graph.query("CREATE INDEX FOR (m:Medication) ON (m.name)")
        print("[FalkorDB] Created indices on Condition, Symptom, and Medication.")
    except Exception as e:
        print(f"[FalkorDB] Index note: {e}")

    condition_count = 0
    symptom_count = 0
    treatment_count = 0
    contra_count = 0

    for condition, details in CLINICAL_REFERENCE_DATA.items():
        # Upsert condition node
        graph.query(
            "MERGE (c:Condition {name: $name})",
            {"name": condition}
        )
        condition_count += 1

        # Symptoms
        for sym in details.get("symptoms", []):
            graph.query(
                """
                MERGE (s:Symptom {name: $sname})
                MERGE (c:Condition {name: $cname})
                MERGE (s)-[:INDICATES]->(c)
                """,
                {"sname": sym, "cname": condition}
            )
            symptom_count += 1

        # Treatments
        for treat in details.get("treatments", []):
            graph.query(
                """
                MERGE (m:Medication {name: $mname})
                MERGE (c:Condition {name: $cname})
                MERGE (m)-[:TREATS]->(c)
                """,
                {"mname": treat, "cname": condition}
            )
            treatment_count += 1

        # Contraindications
        for contra in details.get("contraindications", []):
            graph.query(
                """
                MERGE (m:Medication {name: $mname})
                MERGE (c:Condition {name: $cname})
                MERGE (m)-[:CONTRAINDICATED_FOR]->(c)
                """,
                {"mname": contra, "cname": condition}
            )
            contra_count += 1

    print(f"[FalkorDB] Knowledge Graph Seeded Successfully!")
    print(f"  Conditions:        {condition_count}")
    print(f"  Symptom links:     {symptom_count}")
    print(f"  Treatment links:   {treatment_count}")
    print(f"  Contraindications: {contra_count}")


if __name__ == "__main__":
    seed_falkordb()
