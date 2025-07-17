from GlinerJina import ManualGLiNER
from gliner import GLiNER

# Instantiate model
#model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")


# Load model directly
from transformers import AutoModel
model = GLiNER.from_pretrained("Thehunter99/gliner-fn-base")

# Test text and label set
text= "[Experience with cryopreserved homografts used in aortofemoral position in infection of vascular prostheses].  The authors share herein their experience in reconstructive interventions on the aortofemoral segment in infection of the implant in three 59-to-69-year-old male patients. Infection of the prosthesis was diagnosed by the clinical data, findings of MSCT angiography and duplex scanning of the infrarenal portion of the aorta and arteries of lower limbs. Pseudoaneurysms of distal anastomoses were revealed in two cases. All secondary reconstructions were performed with the use of a cryopreserved aortic bifurcation homograft in the in situ position with simultaneous removal of the infected implant. The results of inoculation from the removed implants yielded Staphylococcus aureus in two cases and Staphylococcus epidermidis in one case. One patient died 6 months later due to causes not related to the operative intervention, in the remaining two cases during one year no findings suggesting reinfection or steno-occlusive lesion of the aortofemoral segment have been revealed."

labels = ["device", "apparatus", "equipment"]



#text = 
"""
Untreated maternal syphilis and adverse outcomes of pregnancy: a systematic review and meta-analysis.
OBJECTIVE  To perform a systematic review and meta-analysis of reported estimates of adverse pregnancy outcomes among untreated women with syphilis and women without syphilis.
METHODS
PubMed, EMBASE and Cochrane Libraries were searched for literature assessing adverse pregnancy outcomes among untreated women with seroreactivity for Treponema pallidum infection and non-seroreactive women.
Adverse pregnancy outcomes were fetal loss or stillbirth, neonatal death, prematurity or low birth weight, clinical evidence of syphilis and infant death.  Random-effects meta-analyses were used to calculate pooled estimates of adverse pregnancy outcomes and, where appropriate, heterogeneity was explored in group-specific analyses.
FINDINGS
Of the 3258 citations identified, only six, all case-control studies, were included in the analysis.
Pooled estimates showed that among untreated pregnant women with syphilis, fetal loss and stillbirth were 21% more frequent, neonatal deaths were 9.3% more frequent and prematurity or low birth weight were 5.8% more frequent than among women without syphilis.
Of the infants of mothers with untreated syphilis, 15% had clinical evidence of congenital syphilis.
The single study that estimated infant death showed a 10% higher frequency among infants of mothers with syphilis.
Substantial heterogeneity was found across studies in the estimates of all adverse outcomes for both women with syphilis (66.5% [95% confidence interval, CI: 58.0-74.1]; I(2) = 91.8%; P < 0.001) and women without syphilis (14.3% [95% CI: 11.8-17.2]; I(2) = 95.9%; P < 0.001).
CONCLUSION
Untreated maternal syphilis is associated with adverse pregnancy outcomes.
These findings can inform policy decisions on resource allocation for the detection of syphilis and its timely treatment in pregnant women.
"""

#labels = ["A person is human being, a celebrity or a public individual", "A prize received for an accomplishment", "A group of players having the same goal and playing together against other"]
#labels = [ "Any deviation from the normal state of an organism: diseases, symptoms, dysfunctions, organ abnormalities (excluding injuries or poisoning).",
''' "Chemical substances, including legal/illegal drugs and biomolecules.",
      "Manufactured object used for medical or laboratory purposes.",
    "Testing of body substances and other diagnostic procedures such as ultrasonography.",
     "Biological function or process in an organism, including organism attributes (e.g., temperature), excluding mental processes.",
      "Organs, body parts, cells, cellular components and body substances.",
     "Statement conveying the results of a scientific observation or experiment.",]'''


# Run prediction
entities = model.predict(text, labels,threshold=0.5)

# Print output
for ent in entities:
    print(f"{ent['text']} => {ent['label']} ( start={ent['start']}, end={ent['end']})")
