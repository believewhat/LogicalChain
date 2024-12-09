import json
import ipdb

import json
import ipdb
import glob
import os
import ipdb
import os
import json
import csv
import pandas as pd
import argparse
import base64
from openai import OpenAI
os.environ["OPENAI_API_KEY"]=''

client = OpenAI(
  api_key=''
)

Prompt = """
Question:
A female newborn delivered at 37 weeks\u2019 gestation develops respiratory distress immediately after birth. She was delivered vaginally to a 31-year-old woman, gravida 1, para 1. Pregnancy was complicated by gestational diabetes mellitus treated with insulin during the third trimester. The newborn's pulse is 136/min, respirations are 57/min, and blood pressure is 60/35 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 91% when the newborn is crying and a saturation of 85% at rest. Examination shows grunting breath sounds and perioral blue discoloration that improves when the patient cries. Lungs are clear to auscultation. Cardiac examination shows a split S2 during inspiration but no murmurs are heard. Femoral pulses are palpable bilaterally. A suction catheter cannot be passed through the nares. In addition to establishing an oral airway, which of the following is the most appropriate treatment for this patient's condition?\nA\nArterial switch procedure\nB\nEndoscopic resection of the posterior nasal septum\nC\nReconnection of the upper esophageal pouch with the lower esophagus\nD\nAnastomosis between subclavian and pulmonary artery\nE\nEndotracheal administration of artificial surfactant
The Correct Answer: B
The Reason:
Correct Answer: B\nEndoscopic resection of the posterior nasal septum\nBilateral choanal atresia, a congenital condition characterized by complete blockage of the nasal cavity, typically manifests immediately after birth with respiratory distress, grunting breath sounds, and central cyanosis which improves with crying and worsens with feeding and at rest. The best initial test is the passage of a catheter through the nares, which is not possible in bilateral choanal atresia. Contrast rhinography in the supine position or CT scan of the facial bones confirms the diagnosis. After establishing an oropharyngeal airway, transnasal endoscopic resection of the posterior nasal septum is the most appropriate treatment to provide a permanent nasal airway. These procedures help prevent life-threatening cyanotic attacks and aspiration when the newborn's mouth is blocked during feeding. Finally, the newborn should be examined for other congenital anomalies since choanal atresia is often associated with CHARGE syndrome.\n\nSplitting of the second heart sound, as seen in this newborn, occurs physiologically during inspiration, which leads to delayed closure of the pulmonary valve (P2) due to increased right ventricular ejection time.\nChoanal atresia\nGIVE FEEDBACK\nArterial switch procedure is indicated in patients with transposition of the great vessels (TGV). TGV is associated with maternal diabetes and manifests with respiratory distress and cyanosis in the newborn immediately after birth, both of which are seen here. In patients with TGV, however, oxygenation is not affected by crying. Furthermore, this condition manifests with extreme cyanosis, progressive hypoxemia, and a loud, single second heart sound on cardiac auscultation.\nBilateral choanal atresia, a congenital condition characterized by complete blockage of the nasal cavity, typically manifests immediately after birth with respiratory distress, grunting breath sounds, and central cyanosis which improves with crying and worsens with feeding and at rest. The best initial test is the passage of a catheter through the nares, which is not possible in bilateral choanal atresia. Contrast rhinography in the supine position or CT scan of the facial bones confirms the diagnosis. After establishing an oropharyngeal airway, transnasal endoscopic resection of the posterior nasal septum is the most appropriate treatment to provide a permanent nasal airway. These procedures help prevent life-threatening cyanotic attacks and aspiration when the newborn's mouth is blocked during feeding. Finally, the newborn should be examined for other congenital anomalies since choanal atresia is often associated with CHARGE syndrome.\n\nSplitting of the second heart sound, as seen in this newborn, occurs physiologically during inspiration, which leads to delayed closure of the pulmonary valve (P2) due to increased right ventricular ejection time.\nChoanal atresia\nReconnection of the upper esophageal pouch with the lower esophagus is indicated in patients with esophageal atresia (EA). This condition may manifest immediately after birth with respiratory distress and cyanotic attacks, similar to what is seen in this patient. Since newborns with EA are unable to swallow properly, excessive foamy secretions at the mouth would also be expected. Moreover, in EA, a gastric catheter can be inserted through the nares but would be blocked at the level of the esophagus.\nAnastomosis between the subclavian and pulmonary artery is performed in newborns with ductal-dependent congenital heart diseases, such as tetralogy of Fallot (TOF), when early heart surgery is not possible. Symptoms of TOF depend on the severity of the right ventricular outflow tract obstruction (RVOTO) and may fluctuate, similar to the fluctuating cyanosis seen in this infant. However, infants with TOF and episodic cyanosis (i.e., tet spells) typically have mild RVOTO and only become symptomatic at 2\u20134 months of age. In contrast, newborns with TOF and cyanosis immediately after birth usually have severe RVOTO with symptoms that do not fluctuate or improve with crying. Finally, cardiac examination would show a harsh systolic ejection murmur at the left upper sternal border and a single second heart sound.\nEndotracheal administration of artificial surfactant is indicated in patients with neonatal respiratory distress syndrome (NRDS), a condition caused by surfactant deficiency in preterm infants. NRDS leads to diffuse atelectasis with respiratory distress and cyanosis after birth, both of which do not improve with crying. Furthermore, NRDS typically manifests with decreased breath sounds on auscultation. This infant was born at term with mature lungs; surfactant would not improve her symptoms.
Question:
A previously healthy 38-year-old man comes to the emergency department for evaluation of several hours of severe, right-sided, colicky flank pain that radiates to the groin. He drinks 2\u20133 energy drinks daily. An x-ray of the abdomen shows a 4-mm radiopaque mass in the right ureter. Microscopic examination of the urine shows biconcave, dumbbell-shaped crystals. Which of the following is most likely to prevent future episodes of flank pain?\nA\nBegin loop diuretic\nB\nMaintain acidic urine\nC\nMaintain adequate hydration\nD\nStart high-dose vitamin C\nE\nRestrict dietary calcium intake\nF\nDecrease caffeine consumption
The Correct Answer: C
The Reason:
Correct Answer: C\nMaintain adequate hydration\nInadequate hydration causes supersaturation of solutes in urine, which increases the risk of stone formation. Fluid intake of \u2265 2.5 L/day reduces the risk of recurrent nephrolithiasis because it decreases the urinary solute concentration and increases the frequency of urination. Adequate hydration prevents recurrence of all types of urinary stones. This patient should also be advised against excessive consumption of energy drinks, which often contain oxalate and contribute to the formation of calcium oxalate stones.\n\nCalcium oxalate stones can also appear as bipyramidal envelopes on urinalysis.\nNephrolithiasis\nGIVE FEEDBACK\nLoop diuretics inhibit reabsorption of calcium in the loop of Henle, causing hypercalciuria, which promotes calcium stone formation. Thiazide diuretics, however, increase calcium reabsorption in the renal tubules and decrease urinary calcium levels, which prevents the formation of calcium stones.\nCalcium oxalate crystallizes in an acidic environment and is soluble in an alkaline environment. In a patient with calcium oxalate stones, alkalinization of urine would prevent recurrent stone formation. This can be achieved with a diet rich in fruits and vegetables or potassium citrate. Urinary acidification is recommended for prevention of calcium phosphate or struvite stones.\nInadequate hydration causes supersaturation of solutes in urine, which increases the risk of stone formation. Fluid intake of \u2265 2.5 L/day reduces the risk of recurrent nephrolithiasis because it decreases the urinary solute concentration and increases the frequency of urination. Adequate hydration prevents recurrence of all types of urinary stones. This patient should also be advised against excessive consumption of energy drinks, which often contain oxalate and contribute to the formation of calcium oxalate stones.\n\nCalcium oxalate stones can also appear as bipyramidal envelopes on urinalysis.\nNephrolithiasis\nVitamin C is metabolized to oxalate, which is then excreted in urine. High doses of supplemental vitamin C cause oxalate supersaturation in urine, which promotes the formation of calcium oxalate stones. Starting this patient on supplemental vitamin C is therefore not the best course of action.\nRestricting normal dietary calcium intake (approx. 1000 mg/day) enhances the intestinal absorption of oxalate. This increases urinary oxalate excretion and the risk of recurrent calcium oxalate stones. In addition, prolonged low dietary calcium can decrease bone mineral density and cause osteoporosis. However, supplemental calcium intake (above the normal daily requirements) should be restricted in patients with calcium stones.\nAlthough the association between caffeinated beverages and formation of renal stones is not entirely clear, increased (not decreased) caffeine consumption (either coffee or tea) decreases the recurrence rate of nephrolithiasis, most likely by improving urinary flow and decreasing urinary supersaturation. Coffee and tea also have antioxidant and anti-inflammatory properties that potentially reduce the risk of stone formation.
"""
"""
files = ['usmle_train_ir.json']
for file_name in files:
    with open(file_name, "r") as file:
        data = json.load(file)

for i in range(len(data)):
    with open(f"./usmle_save_temp/{i}.json", "w") as file:
        json.dump(data[i], file, indent=4)
ipdb.set_trace()
import concurrent.futures
"""
def process_data(x, client):
    q = x['input']
    aw = x['output']
    cnt = 0
    while cnt <5:
        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are a doctor now you should answer the following question."},
                    {"role": "user", "content": f"Here is an example answer template: {Prompt}.\nQuestion:\n{q}\nThe Correct Answer:\n{aw}\nThe Reason:"}
                ]
            )
        except:
            continue
        result = response.choices[0].message.content
        x['output_reason'] = result
        return x
    return x
    
def main():
    parser = argparse.ArgumentParser(description='index')
    parser.add_argument('--index', type=int, default = None)
    args = parser.parse_args()
    with open(f'usmle_save_temp/{args.index}_reason.json', 'r') as file:
        data = json.load(file)
    if len(data['output_reason'].split()) > 10:
        return
    print(data['output_reason'])
    result = process_data(data, client)
    with open(f'usmle_save_temp/{args.index}_reason.json', 'w') as file:
        json.dump(result, file, indent=4)


main()
