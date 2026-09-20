const ROLES=[{id:"researcher",label:"Researcher",hint:"Testable claim, falsifier, rung."},{id:"clinician",label:"Clinician / trialist",hint:"Intent and evidence class — not a home protocol."},{id:"engineer",label:"Engineer / modeller",hint:"What may enter an equation — and what must not."},{id:"builder",label:"Builder / founder",hint:"The bottleneck an organisation can attack."},{id:"student",label:"Student",hint:"The map before the molecule."},{id:"family",label:"Patient advocate / family",hint:"Better questions for a clinical team, not a home protocol."}];
const CANCERS=[{id:"all",name:"Childhood ALL",family:"haematologic",unit:"Often a cure-aimed protocol when the regimen can be delivered. Five-year disease-free survival near 90% in many series is why cure is a fair word here — after time, not at diagnosis.",system:"Clonal marrow disease with MRD as the control signal. Protocolised multi-agent therapy plus supportive care cured most children, not a single target page.",trap:"Do not export ALL success to solid tumours.",states:"Not the default 15-D epithelial TME sketch. Do not pretend T_s/T_r/S_fib is ALL.",example:"An OnCo ALL record is Knowledge. MRD kinetics from a named trial are Evidence."},{id:"testis",name:"Testicular germ-cell tumour",family:"solid-curative",unit:"High probability of clearance with cisplatin-based regimens even when spread.",system:"Chemosensitive clone plus decades of dose and salvage craft.",trap:"This does not prove we will cure cancer. It proves one biology accepted one class of damage.",states:"Do not invent a U slot for cisplatin from an OnCo page.",example:"Cisplatin pages are Knowledge. Trial outcomes are Evidence. They do not set p_lactate."},{id:"tnbc",name:"Triple-negative breast cancer",family:"solid-adaptive",unit:"Localised: multimodality with curative intent. Metastatic: control, adaptation, residual clones.",system:"Heterogeneous clones in a metabolic and immune microenvironment. Lactate, TGF-β, exclusion, persisters are coupled loops.",trap:"LDHA on OnCo is not p_lactate. Legacy LDHA → pyruvate_to_lactate (+0.10) is assumed, not identified.",states:"v2 may annotate L / p_lactate / mct1. It must not write Theta.",example:"bind('ldha') annotates. refuse_knowledge_as_parameter returns forbidden."},{id:"pdac",name:"Pancreatic ductal adenocarcinoma",family:"stroma-late",unit:"Usually control and interception. Late diagnosis is the dominant systems failure.",system:"Dense stroma, few early signals. Detection + resectability + stroma + cachexia + trial access.",trap:"More OnCo targets on the RHS will not move median diagnosis earlier.",states:"S_fib and C_tgfb are the honest lumped handles in v2.",example:"A stage-at-diagnosis idea is Knowledge, not a MechanismObject."},{id:"nsclc",name:"Non-small-cell lung cancer",family:"driver-then-resistance",unit:"Driver subsets: deep control, then resistance. Prevention still removes more deaths than any simulator.",system:"Driver → on-target drug → bypass → new clone. Checkpoint is another branch.",trap:"A pairing page is not a U(t) programme.",states:"T_r is the lumped resistance slot.",example:"EGFR/ALK pages are Knowledge. Randomised outcomes are Evidence."},{id:"gbm",name:"Glioblastoma",family:"cns-adaptive",unit:"Control and time. One clearance is the wrong score.",system:"Spatial evolution inside an organ that cannot be widely resected.",trap:"Fly connectome stubs do not imply a brain-tumour controller.",states:"Do not import ROS from tnbc_mod_3s without audit.",example:"A phase 3 failure is Evidence of a killed idea. Keep it visible."},{id:"crc",name:"Colorectal cancer",family:"split-biology",unit:"Early: surgery-led cure is common. Metastatic MSI-high versus MSS are different diseases.",system:"Screening plus two immunologic worlds. Lumping them into one X is how models lie.",trap:"An MSI idea does not transfer to MSS as a Mechanism.",states:"MSI-high as an I_act context flag remains assumed.",example:"Works-better-together prose is not a coupling."},{id:"melanoma",name:"Melanoma",family:"immuno-adaptive",unit:"A subset sees durable control on checkpoint blockade. That does not generalise to immune-cold carcinomas.",system:"Antigenicity + T-cell fitness + exhaustion (I_act → I_exh).",trap:"PD-1 as an OnCo target is not an identified anti_pd1 U slot.",states:"Annotate only. Wired is not supported.",example:"Checkpoint outcomes are Evidence, not a clinical claim on this site."},{id:"cervix",name:"Cervical cancer",family:"preventable",unit:"HPV vaccine and screening are the dominant systems win. Early invasive disease is often cure-aimed.",system:"Infectious cause plus a screening network.",trap:"Do not hide a vaccine-and-screen problem inside a TME integrator.",states:"Out of scope for confluence_v2_15d as a primary object.",example:"Coverage data are Evidence. Neither belongs in rhs_cancer."},{id:"cml",name:"CML",family:"chronic-control",unit:"Chronic control with TKIs; treatment-free remission is a later subset goal.",system:"One dominant driver, an oral drug, residual stem-like cells.",trap:"CML is not proof that every cancer has a BCR::ABL waiting.",states:"Do not squeeze CML into the TNBC TME vector.",example:"TFR trial data are Evidence. A stopping rule is a Prediction only after that ladder."}];
const SETTINGS=[{id:"prevention",label:"Prevention / interception",hint:"No tumour yet, or only risk."},{id:"local",label:"Localised / early",hint:"Curative intent may be in play."},{id:"metastatic",label:"Metastatic / relapsed",hint:"A population of clones across sites."},{id:"heme",label:"Systemic haematologic",hint:"Marrow or lymphoid field."}];
const STUCK=[{id:"unknown",label:"Biology still unnamed",hint:"Which loop is load-bearing?"},{id:"undruggable",label:"Target known, not reachable",hint:"Chemistry, delivery, site."},{id:"resistance",label:"Adaptation / persisters",hint:"First hit worked. The population changed."},{id:"late",label:"Found too late",hint:"Detection failed first."},{id:"model",label:"Preclinical does not travel",hint:"Models do not carry the human system."},{id:"access",label:"Access, trial, or manufacturing",hint:"Biology is not the binding constraint."}];
const ROLE_NEXT={researcher:"Write one HypothesisObject with a falsifier and a required rung.",clinician:"Name the intent (cure vs control) and the evidence class. Do not import a simulator score as a plan.",engineer:"Keep lineages frozen. Annotate, do not write Theta.",builder:"OnCo startup-request pages are a shelf, not a TAM model.",student:"Memorise the five-layer sentence before any molecule name.",family:"Ask the clinical team the intent (cure vs control). Do not import a simulator score."};
const STUCK_MOVE={unknown:"Stay at Evidence. Fund measurement, not another RHS term.",undruggable:"Chemistry, delivery, or a neighbour — not add-the-gene-to-X.",resistance:"Model a population, not a mean cell.",late:"A better ODE will not find the patient sooner.",model:"Cell-line fit is not clinical Prediction.",access:"Manufacturing and trial-design ideas are not mechanisms."};
const SETTING_NOTE={prevention:"Tilt toward interception and coverage. Do not force a tumour ODE.",local:"Curative intent may be legitimate. Still separate Knowledge from the plan a team runs.",metastatic:"Think in populations and time.",heme:"Borrowing epithelial TME symbols is usually a category error."};
let answers={role:null,cancer:null,setting:null,stuck:null},step=0;
const steps=[{key:"role",title:"Who is asking?",help:"The same cancer needs different next questions depending on who holds the problem.",items:ROLES},{key:"cancer",title:"Which disease — not cancer?",help:"The unit is specific.",items:CANCERS.map(c=>({id:c.id,label:c.name,hint:c.family}))},{key:"setting",title:"What is the current regime?",help:"Intent changes with stage.",items:SETTINGS},{key:"stuck",title:"Where is the system stuck?",help:"Name the binding constraint.",items:STUCK}];
function renderStep(){if(step>=4){renderBoard();return;}document.getElementById("board").hidden=true;document.getElementById("ask").hidden=false;const s=steps[step];document.getElementById("progress").textContent="Question "+(step+1)+" of 4";document.getElementById("qtitle").textContent=s.title;document.getElementById("qhelp").textContent=s.help;const box=document.getElementById("choices");box.innerHTML="";s.items.forEach(item=>{const b=document.createElement("button");b.type="button";b.className="choice";b.innerHTML=item.label+"<small>"+item.hint+"</small>";b.onclick=()=>{answers[s.key]=item.id;step+=1;document.getElementById("backrow").hidden=false;renderStep();};box.appendChild(b);});}
const RESEARCH_DISCLAIMER="Project Confluence disease profiles are computational research artefacts. They are not a medical device, not clinical decision support, not personalized medicine as clinical CDS, and not a claim of cure, diagnosis, or dosing. In-silico / research only. See DISCLAIMER.md.";
const ASKER_ROLE={researcher:"researcher",clinician:"clinician",student:"student",family:"patient_advocate",patient_advocate:"patient_advocate",engineer:"other",builder:"other",other:"other"};
const NON_PARAMETERS=["OnCo knowledge records (pages, targets, ideas)","OnCo confidence.probability","OnCo Idea maturity","Legacy validation/gene_to_parameter_map.json values (including LDHA → pyruvate_to_lactate)","CONFLUENCE v2 symbols such as p_lactate unless independently identified","Auditor classification scores","Grok evidence-audit output","Thinking-lab role, setting, or stuck labels","Clinical intent, cure language, or dosing"];
const CORE_CITATIONS=[
  {id:"1",text:"Sung H, Filho AM, Laversanne M, et al. Global cancer statistics 2024. CA Cancer J Clin. 2026.",url:"https://doi.org/10.3322/caac.70090",doi:"10.3322/caac.70090"},
  {id:"2",text:"World Health Organization. Cancer fact sheet. 2026.",url:"https://www.who.int/news-room/fact-sheets/detail/cancer"},
  {id:"3",text:"National Cancer Institute. Triple-negative breast cancer.",url:"https://www.cancer.gov/types/breast/patient/triple-negative-brochure"},
  {id:"4",text:"Gomila J, OnCo contributors. OnCo: a public, cited knowledge graph of oncology. 2026. Data CC BY-NC 4.0.",url:"https://onco.cc"},
  {id:"5",text:"Ogbonna K. Project Confluence (feat/onco-adapter-p0, pull request #9). 2026.",url:"https://github.com/cloudynirvana/project-confluence/pull/9"},
  {id:"6",text:"Project Confluence. DISCLAIMER.md. Medical non-claims.",url:"https://github.com/cloudynirvana/project-confluence/blob/main/DISCLAIMER.md"},
  {id:"7",text:"Altrock PM, Liu LL, Michor F. The mathematics of cancer: integrating quantitative models. Nat Rev Cancer. 2015.",url:"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5663316/"}
];
const GATES=["Knowledge≠Evidence","Evidence≠Mechanism","Mechanism≠Parameter","Parameter≠Prediction","not_clinical_outcome"];
let lastProfile=null;
function isoNow(){return new Date().toISOString().replace(/\.\d{3}Z$/,"Z");}
function parameterLeak(text){return /p_lactate|pyruvate_to_lactate|confidence\.probability|idea\s*maturity|\btheta\b|Θ|ode\s*parameter|identified parameter/i.test(text);}
function ldhaOncoAsParameter(text){return /\b(ldha|onco)\b/i.test(text)&&(parameterLeak(text)||/parameter/i.test(text));}
function clinicalOverclaim(text){return /\b(cure|dosing|dose|prescribe|personalized medicine|clinical cds|treat this patient|regimen)\b/i.test(text)&&!/\bnot\b/i.test(text);}
function admitCandidate(c){
  const failures=[];
  if(ldhaOncoAsParameter(c.statement)||parameterLeak(c.statement)){failures.push("Mechanism≠Parameter");failures.push("Knowledge≠Evidence");}
  if(!(c.falsifier||"").trim()) failures.push("Evidence≠Mechanism");
  if(clinicalOverclaim(c.statement)){failures.push("not_clinical_outcome");failures.push("Parameter≠Prediction");}
  if(c.evidence_class==="knowledge"||c.evidence_class==="onco_page"||c.evidence_class==="confidence") failures.push("Knowledge≠Evidence");
  if(failures.length) return null;
  return {statement:c.statement,layer:"hypothesis",evidence_class:c.evidence_class,citation_ids:c.citation_ids||[],falsifier:c.falsifier,gates_passed:GATES.slice(),parameter_status:"forbidden_to_enter_theta"};
}
function buildDiseaseProfile(){
  const ca=CANCERS.find(c=>c.id===answers.cancer),role=ROLES.find(r=>r.id===answers.role),setting=SETTINGS.find(s=>s.id===answers.setting),stuck=STUCK.find(s=>s.id===answers.stuck);
  const poison={statement:"OnCo LDHA knowledge and confidence.probability can be entered as the confluence_v2_15d parameter p_lactate (or legacy pyruvate_to_lactate).",evidence_class:"knowledge",citation_ids:["4"],falsifier:"Rejected a priori: Knowledge is not a parameter."};
  const coupled={statement:ca.id==="tnbc"?"Lactate metabolism, TGF-β stroma signalling, immune exclusion and persister states are coupled loops in TNBC and may be stated as a testable mechanism hypothesis.":("In "+ca.name+", clones, microenvironment, and treatment pressure are coupled over time; a single OnCo page does not specify the coupling."),evidence_class:"review_level",citation_ids:ca.id==="tnbc"?["3","7"]:["5","7"],falsifier:"If a named independent measurement shows the putative coupling is absent, retire the hypothesis. Never promote it to p_lactate or any other Θ."};
  const candidates=[coupled,poison];
  const admitted=candidates.map(admitCandidate).filter(Boolean);
  const observables=ca.id==="tnbc"?[
    {statement:"NCI describes triple-negative breast cancer as about 15% of breast cancers and notes faster growth and higher recurrence than some other invasive subtypes. This is a descriptor, not a CONFLUENCE parameter.",citation_ids:["3"]},
    {statement:"OnCo lists LDHA among lactate-metabolism targets. That record is knowledge with CC BY-NC 4.0 attribution, not identified p_lactate.",citation_ids:["4","5"]}
  ]:[{statement:ca.name+" is treated here as a disease-specific systems object. Thinking-lab answers are scaffolds, not measurements.",citation_ids:["5","6"]}];
  return {
    profile_id:"dp-"+ca.id+"-"+isoNow().replace(/[-:]/g,""),
    disease_id:ca.id,
    disease_label:ca.name,
    created_at:isoNow(),
    schema_version:"1.0.0",
    asker_role:ASKER_ROLE[role.id]||"other",
    answers:{
      role:{question:"Who is asking?",choice_id:role.id,choice_label:role.label},
      cancer:{question:"Which disease — not cancer?",choice_id:ca.id,choice_label:ca.name},
      setting:{question:"What is the current regime?",choice_id:setting.id,choice_label:setting.label},
      stuck:{question:"Where is the system stuck?",choice_id:stuck.id,choice_label:stuck.label}
    },
    observables:observables,
    candidate_mechanisms:candidates,
    non_parameters:NON_PARAMETERS.slice(),
    admitted_hypotheses:admitted,
    citations:CORE_CITATIONS.map(function(c){return Object.assign({},c);}),
    disclaimer:RESEARCH_DISCLAIMER
  };
}
function profileFilename(profile){return "disease-profile-"+String(profile.disease_id).toLowerCase().replace(/[^a-z0-9]+/g,"-")+".json";}
function downloadProfile(profile){
  const blob=new Blob([JSON.stringify(profile,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download=profileFilename(profile);
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function(){URL.revokeObjectURL(a.href);},1500);
}
function presentProfile(){
  lastProfile=buildDiseaseProfile();
  const box=document.getElementById("profile-export");
  const summary=document.getElementById("profile-summary");
  if(box) box.hidden=false;
  if(summary){
    summary.innerHTML="<p><strong>"+lastProfile.disease_label+"</strong> · asker: "+lastProfile.asker_role+"</p><p>Admitted hypotheses: "+lastProfile.admitted_hypotheses.length+" (none are ODE parameters). Candidate mechanisms kept for audit: "+lastProfile.candidate_mechanisms.length+". Citations: "+lastProfile.citations.length+" Vancouver entries.</p><p class=\"ask\">OnCo LDHA / confidence.probability / Idea maturity stay in <em>non_parameters</em>. This file is a research artefact — not a protocol, dose, or clinical decision.</p>";
  }
  try{downloadProfile(lastProfile);}catch(err){/* auto-download may be blocked; button remains */}
}
function renderBoard(){const ca=CANCERS.find(c=>c.id===answers.cancer),role=ROLES.find(r=>r.id===answers.role),setting=SETTINGS.find(s=>s.id===answers.setting),stuck=STUCK.find(s=>s.id===answers.stuck);document.getElementById("ask").hidden=true;document.getElementById("board").hidden=false;document.getElementById("board-title").textContent=ca.name+" · a systems board";document.getElementById("board-body").innerHTML="<p>You asked as <strong>"+role.label+"</strong>, about <strong>"+ca.name+"</strong>, in a <strong>"+setting.label.toLowerCase()+"</strong> regime, stuck at <strong>"+stuck.label.toLowerCase()+"</strong>.</p><h3>1. Name the success unit</h3><p>"+ca.unit+"</p><p class=\"ask\">"+SETTING_NOTE[answers.setting]+"</p><h3>2. Coupled system, not a page</h3><p>"+ca.system+"</p><h3>3. Five layers before any number</h3><div class=\"layers\"><div class=\"layer\"><b>Knowledge</b><span>OnCo record. Cited, dated, not Theta.</span></div><div class=\"layer\"><b>Evidence</b><span>Measurement with provenance. OnCo confidence is not P(H).</span></div><div class=\"layer\"><b>Mechanism</b><span>do(U | context) ⇒ ΔX. Idea prose is not this.</span></div><div class=\"layer\"><b>Parameter</b><span>Symbol in a frozen lineage. Legacy maps stay assumed.</span></div><div class=\"layer\"><b>Prediction</b><span>Simulator output. Never a prescription.</span></div></div><p>"+ca.example+"</p><h3>4. What the frozen model may hold</h3><p>"+ca.states+"</p><h3>5. The trap this disease invites</h3><p>"+ca.trap+"</p><h3>6. Move the bottleneck demands</h3><p>"+STUCK_MOVE[answers.stuck]+"</p><h3>7. Next question — not a treatment</h3><p class=\"ask\">"+ROLE_NEXT[answers.role]+"</p><h3>8. What this page refuses</h3><p>No regimen, no dose, no claim that this will cure "+ca.name+". Some diseases are already solved as protocols. That does not license a universal key.</p>";presentProfile();}
document.getElementById("back").onclick=()=>{if(step>0)step-=1;if(step<4)answers[steps[step].key]=null;document.getElementById("backrow").hidden=step===0;renderStep();};
document.getElementById("reset").onclick=()=>{answers={role:null,cancer:null,setting:null,stuck:null};step=0;document.getElementById("backrow").hidden=true;const box=document.getElementById("profile-export");if(box)box.hidden=true;renderStep();};
document.getElementById("download-profile")?.addEventListener("click",()=>{if(lastProfile)downloadProfile(lastProfile);});
renderStep();
