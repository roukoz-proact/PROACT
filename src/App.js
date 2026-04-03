import { useState, useCallback } from "react";

// ─── Stats primitives ────────────────────────────────────────────────────────

function sampleNormal(mean, sd) {
  const u1 = Math.random(), u2 = Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function mean(arr) { return arr.reduce((s, x) => s + x, 0) / arr.length; }
function variance(arr) {
  const m = mean(arr);
  return arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
}

function normalCDF(z) {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5 * (1 + sign * y);
}

function lgamma(x) {
  const c=[76.18009172947146,-86.50532032941677,24.01409824083091,
    -1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];
  let y=x, tmp=x+5.5;
  tmp -= (x+0.5)*Math.log(tmp);
  let ser=1.000000000190015;
  for(let j=0;j<6;j++){y++;ser+=c[j]/y;}
  return -tmp+Math.log(2.5066282746310005*ser/x);
}

function betaCF(a,b,x){
  const maxIter=200,eps=3e-7;
  let c=1,d=1-(a+b)*x/(a+1);
  d=d===0?1e-30:d; d=1/d; let h=d;
  for(let m=1;m<=maxIter;m++){
    let m2=2*m;
    let aa=m*(b-m)*x/((a+m2-1)*(a+m2));
    d=1+aa*d;d=d===0?1e-30:d;c=1+aa/c;c=c===0?1e-30:c;d=1/d;h*=d*c;
    aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1));
    d=1+aa*d;d=d===0?1e-30:d;c=1+aa/c;c=c===0?1e-30:c;d=1/d;
    const del=d*c; h*=del;
    if(Math.abs(del-1)<eps) break;
  }
  return h;
}

function tPvalue(t, df) {
  if (!isFinite(t) || !isFinite(df)) return 1;
  if (df > 120) return 2*(1-normalCDF(Math.abs(t)));
  const x = df / (df + t*t);
  const a=df/2, b=0.5;
  if(x<=0) return 0; if(x>=1) return 1;
  const lbeta = lgamma(a)+lgamma(b)-lgamma(a+b);
  const front = Math.exp(Math.log(x)*a+Math.log(1-x)*b-lbeta)/a;
  return Math.min(front*betaCF(a,b,x), 1);
}

function welchTest(a, b) {
  const na=a.length, nb=b.length;
  if(na<2||nb<2) return 1;
  const ma=mean(a), mb=mean(b);
  const va=variance(a), vb=variance(b);
  const se = Math.sqrt(va/na + vb/nb);
  if(se===0) return ma===mb ? 1 : 0;
  const t = (ma-mb)/se;
  const df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1)+(vb/nb)**2/(nb-1));
  return tPvalue(t, df);
}

function wilcoxonTest(a, b) {
  const na=a.length, nb=b.length;
  const combined = [...a.map(v=>({v,g:0})),...b.map(v=>({v,g:1}))].sort((x,y)=>x.v-y.v);
  let ranks = new Array(combined.length);
  let i=0;
  while(i<combined.length){
    let j=i;
    while(j<combined.length && combined[j].v===combined[i].v) j++;
    const avgRank = (i+1+j)/2;
    for(let k=i;k<j;k++) ranks[k]=avgRank;
    i=j;
  }
  let W=0;
  combined.forEach((d,idx)=>{ if(d.g===0) W+=ranks[idx]; });
  const U = W - na*(na+1)/2;
  const muU = na*nb/2;
  let tieCorr=0;
  i=0;
  while(i<combined.length){
    let j=i;
    while(j<combined.length && combined[j].v===combined[i].v) j++;
    const t=j-i; tieCorr+=t*t*t-t; i=j;
  }
  const N=na+nb;
  const sigmaU=Math.sqrt((na*nb/12)*(N+1-tieCorr/(N*(N-1))));
  if(sigmaU===0) return 1;
  const z=(U-muU)/sigmaU;
  return 2*(1-normalCDF(Math.abs(z)));
}

function ancovaTest(a, b, bA, bB) {
  const allY=[...a,...b], allX=[...bA,...bB];
  const mx=mean(allX), my=mean(allY);
  const sxx=allX.reduce((s,x)=>s+(x-mx)**2,0);
  const sxy=allX.reduce((s,x,i)=>s+(x-mx)*(allY[i]-my),0);
  const beta=sxx===0?0:sxy/sxx;
  const resA=a.map((y,i)=>y-beta*(bA[i]-mx));
  const resB=b.map((y,i)=>y-beta*(bB[i]-mx));
  return welchTest(resA, resB);
}

function holmReject(pValues, alpha) {
  const m=pValues.length;
  const indexed=pValues.map((p,i)=>({p,i})).sort((a,b)=>a.p-b.p);
  const rejected=new Array(m).fill(false);
  for(let k=0;k<m;k++){
    const adjAlpha=alpha/(m-k);
    if(indexed[k].p<adjAlpha) rejected[indexed[k].i]=true;
    else break;
  }
  return rejected;
}

function runSimulation({ nPerArm, means, sd, nSim, alpha, testMethod, analysisModel, dropoutPct }) {
  const comparisons = [
    { name: "Colchicine", active: 1 },
    { name: "Statin",     active: 2 },
    { name: "Combination",active: 3 },
  ];
  const rejectCounts_unadj = [0,0,0];
  const rejectCounts_holm  = [0,0,0];
  const baselineMean=30, baselineSD=15;

  for(let sim=0; sim<nSim; sim++){
    const effN = Math.max(2, Math.round(nPerArm*(1-dropoutPct/100)));
    const armData = means.map(mu=>({
      change:   Array.from({length:effN}, ()=>sampleNormal(mu, sd)),
      baseline: Array.from({length:effN}, ()=>sampleNormal(baselineMean, baselineSD)),
    }));
    const pValues = comparisons.map(({active})=>{
      const aD=armData[active], cD=armData[0];
      if(analysisModel==="ancova") return ancovaTest(aD.change,cD.change,aD.baseline,cD.baseline);
      if(testMethod==="wilcoxon") return wilcoxonTest(aD.change,cD.change);
      return welchTest(aD.change,cD.change);
    });
    pValues.forEach((p,i)=>{ if(p<alpha) rejectCounts_unadj[i]++; });
    holmReject(pValues,alpha).forEach((r,i)=>{ if(r) rejectCounts_holm[i]++; });
  }

  return comparisons.map((c,i)=>({
    name: c.name,
    difference: means[0]-means[c.active],
    power_unadj: rejectCounts_unadj[i]/nSim,
    power_holm:  rejectCounts_holm[i]/nSim,
  }));
}

// ─── UI ──────────────────────────────────────────────────────────────────────

const mono = "'IBM Plex Mono', monospace";
const sans = "'IBM Plex Sans', system-ui, sans-serif";
const card = { background:"#0d1117", border:"1px solid #21262d", borderRadius:8, padding:"16px 18px", marginBottom:14 };
const secLabel = { fontSize:10, color:"#484f58", fontFamily:mono, letterSpacing:"0.14em", textTransform:"uppercase", borderBottom:"1px solid #21262d", paddingBottom:7, marginBottom:14 };

function Slider({label,value,min,max,step,onChange,unit=""}) {
  return (
    <div style={{marginBottom:12}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
        <span style={{fontSize:12,color:"#8b949e",fontFamily:mono}}>{label}</span>
        <span style={{fontSize:12,color:"#e6edf3",fontFamily:mono}}>{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>onChange(Number(e.target.value))}
        style={{width:"100%",accentColor:"#58a6ff",cursor:"pointer"}}/>
    </div>
  );
}

function Sel({label,value,options,onChange}) {
  return (
    <div style={{marginBottom:12}}>
      <div style={{fontSize:12,color:"#8b949e",fontFamily:mono,marginBottom:4}}>{label}</div>
      <select value={value} onChange={e=>onChange(e.target.value)} style={{
        width:"100%",background:"#161b22",border:"1px solid #30363d",
        color:"#e6edf3",padding:"6px 9px",borderRadius:5,fontSize:12,fontFamily:mono,cursor:"pointer",
      }}>
        {options.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </div>
  );
}

function PowerRow({name,diff,power_unadj,power_holm,nEff}) {
  const fmt=p=>`${Math.round(p*100)}%`;
  const col=p=>p>=0.80?"#3fb950":p>=0.65?"#d29922":"#f85149";
  const badge=p=>p>=0.80?"Adequate":p>=0.65?"Marginal":"Low";
  return (
    <div style={{marginBottom:20}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{fontFamily:mono,fontSize:12,color:"#e6edf3"}}>{name} vs Placebo</span>
        <span style={{fontFamily:mono,fontSize:11,color:"#484f58"}}>Δ={diff.toFixed(1)} mm³ · n={nEff}/arm</span>
      </div>
      {[{label:"Unadj.", power:power_unadj, labelColor:"#8b949e"},
        {label:"Holm",   power:power_holm,  labelColor:"#58a6ff"}].map(({label,power,labelColor})=>(
        <div key={label} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
          <span style={{fontFamily:mono,fontSize:10,color:labelColor,width:52,flexShrink:0}}>{label}</span>
          <div style={{flex:1,background:"#21262d",borderRadius:3,height:8,overflow:"hidden"}}>
            <div style={{height:"100%",width:`${Math.round(power*100)}%`,
              background:col(power),borderRadius:3,transition:"width .5s ease"}}/>
          </div>
          <span style={{fontFamily:mono,fontSize:12,color:col(power),width:40,textAlign:"right",fontWeight:700}}>{fmt(power)}</span>
          <span style={{fontFamily:mono,fontSize:10,color:"#484f58",width:60}}>{badge(power)}</span>
        </div>
      ))}
    </div>
  );
}

const PRESETS = [
  {label:"Base case",               desc:"n=50, SD=17, no dropout, t-test",           s:{nPerArm:50,sd:17,dropoutPct:0,testMethod:"ttest",  analysisModel:"primary",placebo:6.7,colchicine:-4.1,statin:-8.2,combo:-12.3}},
  {label:"SD = 20",                 desc:"Conservative variability assumption",        s:{nPerArm:50,sd:20,dropoutPct:0,testMethod:"ttest",  analysisModel:"primary",placebo:6.7,colchicine:-4.1,statin:-8.2,combo:-12.3}},
  {label:"Reduced colchicine",      desc:"50% smaller effect (Δ≈4.7 mm³), SD=17",    s:{nPerArm:50,sd:17,dropoutPct:0,testMethod:"ttest",  analysisModel:"primary",placebo:6.7,colchicine:2.0, statin:-8.2,combo:-12.3}},
  {label:"Reduced colch + SD=20",   desc:"Most conservative base assumptions",        s:{nPerArm:50,sd:20,dropoutPct:0,testMethod:"ttest",  analysisModel:"primary",placebo:6.7,colchicine:2.0, statin:-8.2,combo:-12.3}},
  {label:"Expansion n=54",          desc:"Reduced colch + SD=20, expanded enroll",    s:{nPerArm:54,sd:20,dropoutPct:0,testMethod:"ttest",  analysisModel:"primary",placebo:6.7,colchicine:2.0, statin:-8.2,combo:-12.3}},
  {label:"10% dropout sensitivity", desc:"Sensitivity only — not SAP base case",      s:{nPerArm:50,sd:17,dropoutPct:10,testMethod:"ttest", analysisModel:"primary",placebo:6.7,colchicine:-4.1,statin:-8.2,combo:-12.3}},
  {label:"Wilcoxon primary",        desc:"Non-parametric test, base assumptions",     s:{nPerArm:50,sd:17,dropoutPct:0,testMethod:"wilcoxon",analysisModel:"primary",placebo:6.7,colchicine:-4.1,statin:-8.2,combo:-12.3}},
  {label:"ANCOVA (supportive)",     desc:"Baseline-adjusted per SAP §4.2.3",          s:{nPerArm:50,sd:17,dropoutPct:0,testMethod:"ttest",  analysisModel:"ancova",  placebo:6.7,colchicine:-4.1,statin:-8.2,combo:-12.3}},
];

export default function App() {
  const [nPerArm,setNPerArm]=useState(50);
  const [sd,setSd]=useState(17);
  const [dropoutPct,setDropoutPct]=useState(0);
  const [alpha,setAlpha]=useState(0.05);
  const [nSim,setNSim]=useState(5000);
  const [testMethod,setTestMethod]=useState("ttest");
  const [analysisModel,setAnalysisModel]=useState("primary");
  const [placebo,setPlacebo]=useState(6.7);
  const [colchicine,setColchicine]=useState(-4.1);
  const [statin,setStatin]=useState(-8.2);
  const [combo,setCombo]=useState(-12.3);
  const [results,setResults]=useState(null);
  const [running,setRunning]=useState(false);
  const [elapsed,setElapsed]=useState(null);
  const [activePreset,setActivePreset]=useState(null);

  const applyState=s=>{
    setNPerArm(s.nPerArm);setSd(s.sd);setDropoutPct(s.dropoutPct);
    setTestMethod(s.testMethod);setAnalysisModel(s.analysisModel);
    setPlacebo(s.placebo);setColchicine(s.colchicine);
    setStatin(s.statin);setCombo(s.combo);setResults(null);
  };

  const run=useCallback(()=>{
    setRunning(true);setResults(null);
    const t0=performance.now();
    setTimeout(()=>{
      const res=runSimulation({nPerArm,means:[placebo,colchicine,statin,combo],
        sd,nSim,alpha,testMethod,analysisModel,dropoutPct});
      setResults(res);
      setElapsed(((performance.now()-t0)/1000).toFixed(2));
      setRunning(false);
    },20);
  },[nPerArm,sd,dropoutPct,alpha,nSim,testMethod,analysisModel,placebo,colchicine,statin,combo]);

  const effN=Math.max(2,Math.round(nPerArm*(1-dropoutPct/100)));

  return (
    <div style={{minHeight:"100vh",background:"#010409",color:"#e6edf3",fontFamily:sans,padding:"22px 18px",maxWidth:960,margin:"0 auto"}}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>

      {/* Header */}
      <div style={{marginBottom:22,borderBottom:"1px solid #21262d",paddingBottom:16}}>
        <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
          <span style={{fontFamily:mono,fontSize:10,color:"#484f58",letterSpacing:"0.16em"}}>PROACT 2</span>
          <span style={{color:"#21262d"}}>·</span>
          <span style={{fontFamily:mono,fontSize:10,color:"#484f58",letterSpacing:"0.12em"}}>SAP-ALIGNED POWER SIMULATION</span>
        </div>
        <h1 style={{fontSize:20,fontWeight:600,margin:0,letterSpacing:"-0.02em",color:"#f0f6fc"}}>Monte Carlo Power Analysis</h1>
        <p style={{margin:"4px 0 0",fontSize:12,color:"#484f58",fontFamily:mono}}>
          Pairwise comparisons vs placebo · Unadjusted + Holm-Bonferroni (SAP §6) · Primary: change score t-test or Wilcoxon (SAP §4.2.3)
        </p>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"310px 1fr",gap:16}}>
        {/* LEFT */}
        <div>
          {/* Scenario library */}
          <div style={card}>
            <div style={secLabel}>Scenario Library</div>
            {PRESETS.map((p,i)=>(
              <div key={i} onClick={()=>{setActivePreset(i);applyState(p.s);}} style={{
                padding:"8px 10px",borderRadius:5,marginBottom:3,cursor:"pointer",
                background:activePreset===i?"#161b22":"transparent",
                border:activePreset===i?"1px solid #30363d":"1px solid transparent",
                transition:"all 0.1s",
              }}>
                <div style={{fontSize:12,fontFamily:mono,color:activePreset===i?"#58a6ff":"#8b949e",marginBottom:1}}>{p.label}</div>
                <div style={{fontSize:11,color:"#484f58",fontFamily:mono}}>{p.desc}</div>
              </div>
            ))}
          </div>

          {/* Means */}
          <div style={card}>
            <div style={secLabel}>Mean ΔNCPV at 12 mo (mm³)</div>
            <Slider label="Placebo"     value={placebo}    min={0}   max={15}  step={0.1} onChange={v=>{setPlacebo(v);setActivePreset(null);}}    unit=" mm³"/>
            <Slider label="Colchicine"  value={colchicine} min={-15} max={10}  step={0.1} onChange={v=>{setColchicine(v);setActivePreset(null);}} unit=" mm³"/>
            <Slider label="Statin"      value={statin}     min={-20} max={10}  step={0.1} onChange={v=>{setStatin(v);setActivePreset(null);}}     unit=" mm³"/>
            <Slider label="Combination" value={combo}      min={-25} max={10}  step={0.1} onChange={v=>{setCombo(v);setActivePreset(null);}}      unit=" mm³"/>
            <div style={{background:"#161b22",borderRadius:5,padding:"9px 11px",marginTop:4}}>
              <div style={{fontSize:10,color:"#484f58",fontFamily:mono,letterSpacing:"0.1em",marginBottom:6}}>DIFFERENCES VS PLACEBO</div>
              {[["Colchicine",colchicine],["Statin",statin],["Combination",combo]].map(([l,v])=>(
                <div key={l} style={{display:"flex",justifyContent:"space-between",fontSize:12,fontFamily:mono,marginBottom:3}}>
                  <span style={{color:"#8b949e"}}>{l}</span>
                  <span style={{color:"#79c0ff"}}>{(placebo-v).toFixed(1)} mm³</span>
                </div>
              ))}
            </div>
          </div>

          {/* Design */}
          <div style={card}>
            <div style={secLabel}>Trial Design (Base Case)</div>
            <Slider label="N per arm (analyzed)" value={nPerArm} min={20} max={80} step={1} onChange={v=>{setNPerArm(v);setActivePreset(null);}}/>
            <div style={{fontSize:11,color:"#484f58",fontFamily:mono,marginBottom:10}}>Total N = {nPerArm*4} randomized</div>
            <Slider label="SD of ΔNCPV" value={sd} min={10} max={30} step={0.5} onChange={v=>{setSd(v);setActivePreset(null);}} unit=" mm³"/>
          </div>

          {/* Statistical framework */}
          <div style={card}>
            <div style={secLabel}>Statistical Framework</div>
            <Sel label="Primary Test  (SAP §4.2.3)" value={testMethod} onChange={v=>{setTestMethod(v);setActivePreset(null);}} options={[
              {v:"ttest",   l:"Two-sample t-test (primary default)"},
              {v:"wilcoxon",l:"Wilcoxon rank-sum (if non-normal)"},
            ]}/>
            <Sel label="Analysis Model" value={analysisModel} onChange={v=>{setAnalysisModel(v);setActivePreset(null);}} options={[
              {v:"primary",l:"Change score — primary (SAP §4.2.3)"},
              {v:"ancova",  l:"ANCOVA — supportive only (SAP §4.2.3)"},
            ]}/>
            {/* Multiplicity — fixed, informational */}
            <div style={{background:"#161b22",border:"1px solid #21262d",borderRadius:5,padding:"9px 11px",marginBottom:12}}>
              <div style={{fontSize:10,color:"#484f58",fontFamily:mono,letterSpacing:"0.1em",marginBottom:5}}>MULTIPLICITY · SAP §6 (FIXED)</div>
              <div style={{fontSize:11,color:"#8b949e",fontFamily:mono,lineHeight:1.65}}>
                Bonferroni-Holm across 3 comparisons.<br/>
                Results show unadjusted and Holm-adjusted power side-by-side.
              </div>
            </div>
            <Sel label="Nominal α" value={String(alpha)} onChange={v=>{setAlpha(Number(v));setActivePreset(null);}} options={[
              {v:"0.05", l:"0.05 (two-sided)"},
              {v:"0.025",l:"0.025"},
            ]}/>
          </div>

          {/* Dropout — sensitivity only */}
          <div style={{...card,border:"1px solid #2d2008"}}>
            <div style={{...secLabel,color:"#6e5b2a",borderBottomColor:"#2d2008"}}>Dropout Sensitivity (not base case)</div>
            <div style={{fontSize:11,color:"#6e5b2a",fontFamily:mono,marginBottom:10,lineHeight:1.5}}>
              SAP base case assumes 0% dropout (§3.2). Use for sensitivity only.
            </div>
            <Slider label="Dropout rate" value={dropoutPct} min={0} max={30} step={1}
              onChange={v=>{setDropoutPct(v);setActivePreset(null);}} unit="%"/>
            <div style={{fontSize:11,color:"#484f58",fontFamily:mono}}>Effective n: {effN}/arm · total {effN*4}</div>
          </div>

          <Sel label="Simulations" value={String(nSim)} onChange={v=>setNSim(Number(v))} options={[
            {v:"1000",l:"1,000  (fast)"},
            {v:"5000",l:"5,000  (recommended)"},
            {v:"10000",l:"10,000 (precise)"},
          ]}/>

          <button onClick={run} disabled={running} style={{
            width:"100%",padding:"11px",
            background:running?"#161b22":"#1f6feb",
            border:running?"1px solid #30363d":"1px solid #388bfd",
            borderRadius:6,color:running?"#484f58":"#f0f6fc",
            fontSize:13,fontWeight:600,cursor:running?"not-allowed":"pointer",
            fontFamily:mono,transition:"all .15s",
          }}>
            {running?`⟳  Running ${nSim.toLocaleString()} iterations…`:"▶  Run Simulation"}
          </button>
        </div>

        {/* RIGHT */}
        <div>
          {!results&&!running&&(
            <div style={{...card,minHeight:280,display:"flex",alignItems:"center",justifyContent:"center",border:"1px dashed #21262d"}}>
              <span style={{fontFamily:mono,fontSize:12,color:"#30363d"}}>Select a scenario and run</span>
            </div>
          )}
          {running&&(
            <div style={{...card,minHeight:280,display:"flex",alignItems:"center",justifyContent:"center"}}>
              <span style={{fontFamily:mono,fontSize:13,color:"#58a6ff"}}>Simulating…</span>
            </div>
          )}
          {results&&(<>
            {/* Main results */}
            <div style={card}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:12}}>
                <div>
                  <div style={secLabel}>Power Results</div>
                  <div style={{fontSize:11,color:"#484f58",fontFamily:mono,marginTop:-8}}>
                    {nSim.toLocaleString()} iterations · {elapsed}s ·{" "}
                    {testMethod==="ttest"?"Welch t-test":"Wilcoxon"} ·{" "}
                    {analysisModel==="ancova"?"ANCOVA":"Change score"} · α={alpha}
                    {dropoutPct>0?` · ${dropoutPct}% dropout`:""}
                  </div>
                </div>
                <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"flex-end"}}>
                  {analysisModel==="ancova"&&(
                    <span style={{fontFamily:mono,fontSize:10,color:"#58a6ff",background:"#0d1f2e",
                      border:"1px solid #0d419d",borderRadius:4,padding:"3px 7px"}}>SUPPORTIVE</span>
                  )}
                  {dropoutPct>0&&(
                    <span style={{fontFamily:mono,fontSize:10,color:"#d29922",background:"#2d1f00",
                      border:"1px solid #6e3e00",borderRadius:4,padding:"3px 7px"}}>SENSITIVITY</span>
                  )}
                </div>
              </div>
              {results.map(r=>(
                <PowerRow key={r.name} name={r.name} diff={r.difference}
                  nEff={effN} power_unadj={r.power_unadj} power_holm={r.power_holm}/>
              ))}
              <div style={{borderTop:"1px solid #21262d",paddingTop:8,marginTop:4,
                fontSize:11,color:"#484f58",fontFamily:mono,lineHeight:1.65}}>
                Holm procedure: comparisons ordered by ascending p-value; sequential α = α/(m−k). Power = % simulations rejecting H₀.
              </div>
            </div>

            {/* Table */}
            <div style={card}>
              <div style={secLabel}>Summary Table</div>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12,fontFamily:mono}}>
                <thead>
                  <tr style={{borderBottom:"1px solid #21262d"}}>
                    {["Comparison","Δ (mm³)","n/arm","SD","Power — unadj","Power — Holm"].map(h=>(
                      <th key={h} style={{textAlign:"left",color:"#484f58",fontWeight:500,paddingBottom:7,paddingRight:14}}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.map(r=>{
                    const pu=Math.round(r.power_unadj*100),ph=Math.round(r.power_holm*100);
                    const cu=pu>=80?"#3fb950":pu>=65?"#d29922":"#f85149";
                    const ch=ph>=80?"#3fb950":ph>=65?"#d29922":"#f85149";
                    return (
                      <tr key={r.name} style={{borderBottom:"1px solid #161b22"}}>
                        <td style={{padding:"7px 14px 7px 0",color:"#e6edf3"}}>{r.name} vs Placebo</td>
                        <td style={{color:"#79c0ff",paddingRight:14}}>{r.difference.toFixed(1)}</td>
                        <td style={{color:"#8b949e",paddingRight:14}}>{effN}</td>
                        <td style={{color:"#8b949e",paddingRight:14}}>{sd}</td>
                        <td style={{color:cu,fontWeight:700,paddingRight:14}}>{pu}%</td>
                        <td style={{color:ch,fontWeight:700}}>{ph}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Interpretation */}
            <div style={{...card,background:"#0a0f1a",border:"1px solid #1a2640"}}>
              <div style={{fontSize:10,color:"#1f6feb",fontFamily:mono,letterSpacing:"0.12em",marginBottom:10}}>INTERPRETATION</div>
              {results.map(r=>{
                const ph=Math.round(r.power_holm*100),pu=Math.round(r.power_unadj*100);
                const msg = ph>=90 ? "Well-powered after Holm correction. Comfortable margin."
                  : ph>=80 ? "Adequately powered after Holm correction. Enroll to target."
                  : ph>=65 ? "Marginal after Holm correction. Revisit effect assumptions or expand enrollment."
                  : "Underpowered after Holm correction. Expansion or revised assumptions required.";
                return (
                  <div key={r.name} style={{marginBottom:10,fontSize:12,lineHeight:1.65}}>
                    <span style={{color:"#58a6ff",fontFamily:mono}}>{r.name} — unadj {pu}% / Holm {ph}%: </span>
                    <span style={{color:"#8b949e"}}>{msg}</span>
                  </div>
                );
              })}
            </div>

            {/* SAP alignment */}
            <div style={{...card,background:"transparent",border:"1px solid #21262d"}}>
              <div style={{fontSize:10,color:"#484f58",fontFamily:mono,letterSpacing:"0.12em",marginBottom:10}}>SAP ALIGNMENT</div>
              <div style={{fontSize:11,color:"#484f58",fontFamily:mono,lineHeight:1.8}}>
                <div>· Primary: pairwise t-test or Wilcoxon on change score (§4.2.3) — not omnibus</div>
                <div>· ANCOVA is supportive, not primary — labeled accordingly</div>
                <div>· Multiplicity: Bonferroni-Holm across 3 active-vs-placebo pairs (§6)</div>
                <div>· Base case: 50 analyzed / arm, no dropout assumed (§3.2)</div>
                <div>· Dropout panel is sensitivity only, separated visually</div>
              </div>
            </div>
          </>)}
        </div>
      </div>
    </div>
  );
}
