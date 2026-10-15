'use strict';
const $=id=>document.getElementById(id),asset=n=>`assets/stages/${n}.webp`;
const focus=[
 ['focus-input','Four reference particle centers','The same physical crop, with the four annotated particle centers in gold.','focus-input','Four reference particle centers','Point annotations define the localization target.'],
 ['focus-mask','Four particles merge into one foreground','The saved 3D semantic foreground is one connected component spanning all four reference particles.','focus-local','The detector predicts particle centers','The local detector produces a labeled point set. This saved detector endpoint contains four correct picks and two false positives.'],
 ['focus-centers','Mean-shift returns two inaccurate centers','Native center extraction produces two positions. Neither matches a reference center within the 3 nm tolerance.','focus-ours','TomoDETR localizes all four particles','The complete system combines direct point prediction and cross-window refinement: four correct picks, zero false positives.'],
 ['focus-final','0 correct · 2 false positives · 4 missed','Final DeepFinder predictions, using the native post-processing and frozen operating point.','focus-ours','4 correct · 0 false positives · 0 missed','Final TomoDETR predictions in exactly the same crop and 3D evaluation tolerance.']
];let stage=1,stageTimer=null;
function showFocus(n){stage=n;const s=focus[n];$('focus-baseline').src=asset(s[0]);$('baseline-stage-title').textContent=s[1];$('baseline-stage-caption').textContent=s[2];$('focus-baseline').alt=s[1];$('focus-direct').src=asset(s[3]);$('direct-stage-title').textContent=s[4];$('direct-stage-caption').textContent=s[5];$('focus-direct').alt=s[4];document.querySelectorAll('button[data-stage]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.stage)===n)));}
function stopFocus(){clearInterval(stageTimer);stageTimer=null;$('stage-play').textContent='▶ Play recorded stages';$('stage-play').setAttribute('aria-pressed','false');}
$('stage-play').onclick=()=>{if(stageTimer){stopFocus();return;}showFocus(0);$('stage-play').textContent='Ⅱ Pause stages';$('stage-play').setAttribute('aria-pressed','true');stageTimer=setInterval(()=>showFocus((stage+1)%4),3000);};document.querySelectorAll('button[data-stage]').forEach(b=>b.onclick=()=>{stopFocus();showFocus(Number(b.dataset.stage));});showFocus(1);
const traces={
 DeepFinder:{result:'Final crop: 0 correct · 3 false positives · 6 missed',note:'The orange overlay isolates the actual 3D semantic component spanning the six reference particles. Its native clustering yields two centers; the final crop contains three predictions in total. This saved semantic labelmap is not a probability map.',stages:[
 ['trace-input','Input & reference','Six reference particles in a crowded apo-ferritin region. The background and coordinates stay fixed throughout the trace.'],
 ['deepfinder-mask','Connected semantic foreground','Six reference centers are assigned to one connected semantic region (5,696 full-resolution foreground voxels). Separate instance identities are not represented in this foreground label.'],
 ['deepfinder-clusters','Native mean-shift centers','The native clustering output contains two centers from this connected region. Their positions do not recover the six individual reference centers.'],
 ['deepfinder-final','Final localized centers','All three final DeepFinder predictions inside the displayed 3D crop are false positives under the 3 nm matching tolerance. All six reference particles are missed.']]},
 DeepETPicker:{result:'Final crop: 1 correct · 3 false positives · 5 missed',note:'The dense maps show the mean of exact patch outputs over overlapping inference cores. Threshold fraction is the fraction of patch masks passing the frozen 0.5 threshold. MP-NMS is computed patch-wise; the displayed final candidates reproduce the saved native outputs.',stages:[
 ['trace-input','Input & reference','The same six-particle region, with reference centers in gold.'],
 ['deepet-prob','Dense probability response','The frozen network predicts a voxel-wise foreground probability. Max projection exposes the spatial support of its response (color scale: 0–1).'],
 ['deepet-threshold','Thresholded patch masks','Each patch probability map is thresholded at 0.5. The visualization shows the fraction of overlapping patch masks voting foreground (0–1), preserving the actual patch-wise operation.'],
 ['deepet-smooth','Repeated mean pooling','The thresholded masks are repeatedly mean-pooled (5-voxel kernel, eight repetitions). This native smoothed response is the input to peak selection. Colors show the recorded response on a 0–1 scale.'],
 ['deepetpicker-final','MP-NMS & final picks','Native maxima selection and duplicate removal retain four candidate centers in the crop. One matches a reference particle; three are false positives, leaving five missed particles.']]},
 OCTOPI:{result:'Final crop: 2 correct · 0 false positives · 4 missed',note:'Probabilities and watershed intermediates come from a replay of the frozen OCTOPI checkpoint and its native post-processing. Final replayed coordinates agree with saved predictions to less than 0.001 Å. The distance-map colors are normalized to its maximum for display.',stages:[
 ['trace-input','Input & reference','The same six-particle crop used for the other baseline traces.'],
 ['octopi-prob','Dense probability response','Foreground probabilities from sliding-window inference and test-time augmentation. Colors show the 0–1 probability scale.'],
 ['octopi-mask','Semantic argmax mask','The voxel-wise class argmax forms the foreground mask. In this region a pair is connected, while another reference particle has no nearby foreground.'],
 ['octopi-open','Morphological opening','Native binary opening with a radius-one ball cleans the semantic foreground before instance extraction.'],
 ['octopi-distance','Distance transform','The distance to the opened-mask boundary is computed in 3D. Its local maxima will seed the watershed.'],
 ['octopi-maxima','Watershed seeds','Native local maxima of the distance field become watershed markers. They are determined by the predicted foreground geometry.'],
 ['octopi-watershed','Watershed regions','The watershed partitions the opened foreground into regions. Colors distinguish the resulting labels in the displayed max projection.'],
 ['octopi-filter','Size-filtered regions','Native region-size criteria retain a subset of watershed regions for centroid extraction.'],
 ['octopi-final','Centroids & final picks','Centroid extraction and native repeated-pick merging leave two correct picks in this crop. Four of the six reference particles remain missed.']]}
};let method='DeepFinder',traceIndex=1,traceTimer=null;
function stopTrace(){clearInterval(traceTimer);traceTimer=null;$('trace-play').textContent='▶ Play this pipeline';$('trace-play').setAttribute('aria-pressed','false');}
function showTrace(n){traceIndex=n;const t=traces[method],s=t.stages[n];$('trace-image').src=asset(s[0]);$('trace-image').alt=`${method}: ${s[1]}`;$('trace-title').textContent=s[1];$('trace-description').textContent=s[2];$('trace-result').textContent=t.result;const scale=/prob|threshold|smooth|distance/.test(s[0]);$('trace-scale').hidden=!scale;$('trace-scale-name').textContent=s[0].includes('distance')?'Normalized distance':s[0].includes('threshold')?'Patch threshold fraction':s[0].includes('smooth')?'Pooled response':'Probability';$('trace-scale').classList.toggle('viridis',/smooth|distance/.test(s[0]));document.querySelectorAll('[data-trace-step]').forEach(b=>b.setAttribute('aria-current',String(Number(b.dataset.traceStep)===n)));}
function selectMethod(name){stopTrace();method=name;document.querySelectorAll('[data-method]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.method===name)));$('trace-footnote').textContent=traces[name].note;$('trace-steps').replaceChildren();$('trace-thumbs').replaceChildren();traces[name].stages.forEach((s,i)=>{let li=document.createElement('li'),b=document.createElement('button');b.textContent=`${String(i+1).padStart(2,'0')} · ${s[1]}`;b.dataset.traceStep=i;b.onclick=()=>{stopTrace();showTrace(i);};li.append(b);$('trace-steps').append(li);let t=document.createElement('button'),im=document.createElement('img'),span=document.createElement('span');t.dataset.traceStep=i;t.setAttribute('aria-label',`${name}: ${s[1]}`);im.src=asset(s[0]);im.alt='';im.width=134;im.height=103;span.textContent=`${i+1}. ${s[1]}`;t.append(im,span);t.onclick=()=>{stopTrace();showTrace(i);};$('trace-thumbs').append(t);});showTrace(1);}
$('trace-play').onclick=()=>{if(traceTimer){stopTrace();return;}showTrace(0);$('trace-play').textContent='Ⅱ Pause pipeline';$('trace-play').setAttribute('aria-pressed','true');traceTimer=setInterval(()=>showTrace((traceIndex+1)%traces[method].stages.length),2600);};document.querySelectorAll('[data-method]').forEach(b=>b.onclick=()=>selectMethod(b.dataset.method));selectMethod(method);
// The 3D inset renders exposed faces of the actual foreground voxels.
let surface=null,angle=.55,tilt=.32,rotating=false,frame=0,previous=0,pointer=null;
const canvas=$('surface'),ctx=canvas.getContext('2d');
function rotate(v){let x=v[0]*Math.cos(angle)+v[2]*Math.sin(angle),z=-v[0]*Math.sin(angle)+v[2]*Math.cos(angle);return [x,v[1]*Math.cos(tilt)-z*Math.sin(tilt),v[1]*Math.sin(tilt)+z*Math.cos(tilt)];}
function project(v){const p=rotate(v.map((x,i)=>x-surface.shape[i]/2));return [350+p[0]*10,210-p[1]*10,p[2]];}
function drawSurface(){if(!surface)return;ctx.fillStyle='#151613';ctx.fillRect(0,0,700,440);let items=surface.faces.map(f=>{let p=f.v.map(project),n=rotate(f.n);return{p,z:p.reduce((v,p)=>v+p[2],0)/4,l:.53+.35*Math.max(0,n[0]*.35+n[1]*.4+n[2]*.8)};});items.sort((a,b)=>a.z-b.z);for(const f of items){ctx.beginPath();f.p.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=`rgb(${Math.round(239*f.l)},${Math.round(164*f.l)},${Math.round(83*f.l)})`;ctx.fill();}for(const [kind,points] of [['gt',surface.gt],['centers',surface.centers]])for(const v of points){const [x,y]=project(v);ctx.beginPath();ctx.lineWidth=3;ctx.strokeStyle=kind==='gt'?'#ffe7a1':'#d62e54';ctx.shadowColor='#182622';ctx.shadowBlur=3;if(kind==='gt')ctx.arc(x,y,10,0,Math.PI*2);else{ctx.moveTo(x-8,y-8);ctx.lineTo(x+8,y+8);ctx.moveTo(x-8,y+8);ctx.lineTo(x+8,y-8);}ctx.stroke();ctx.shadowBlur=0;}ctx.fillStyle='#aaa89f';ctx.font='12px Georgia';ctx.fillText('Saved 3D semantic component · reference centers shown as guides',18,421);}
function stopRotate(){rotating=false;cancelAnimationFrame(frame);previous=0;$('surface-play').textContent='▶ Rotate foreground';$('surface-play').setAttribute('aria-pressed','false');}
function spin(t){if(!rotating)return;if(previous)angle+=Math.min(t-previous,60)*.00032;previous=t;drawSurface();frame=requestAnimationFrame(spin);}
$('surface-play').onclick=()=>{if(!surface)return;if(rotating){stopRotate();return;}rotating=true;previous=0;$('surface-play').textContent='Ⅱ Pause rotation';$('surface-play').setAttribute('aria-pressed','true');frame=requestAnimationFrame(spin);};
$('surface-details').addEventListener('toggle',async()=>{if(!$('surface-details').open){stopRotate();return;}if(!surface){try{const r=await fetch('assets/stages/surface.json');if(!r.ok)throw Error();surface=await r.json();}catch{$('surface-play').textContent='Unable to load. Reopen to retry.';return;}}drawSurface();});
canvas.onpointerdown=e=>{if(!surface)return;stopRotate();pointer=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!pointer)return;angle+=(e.clientX-pointer[0])*.008;tilt=Math.max(-1.3,Math.min(1.3,tilt+(e.clientY-pointer[1])*.008));pointer=[e.clientX,e.clientY];drawSurface();};canvas.onpointerup=canvas.onpointercancel=()=>pointer=null;
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopRotate();}});
// Visible players loop automatically; scrubbing pauses at an exact recorded stage.
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
function makePlayer({section,range,output,button,count,index,show}){
 let playing=!reducedMotion.matches,visible=false,position=index(),last=0;
 const slider=$(range),label=$(output),control=$(button),duration=2100;
 function paint(){const n=count();slider.max=n-1;slider.value=Math.min(position,n-1);label.textContent=`${index()+1} / ${n}`;control.textContent=playing?'Ⅱ Pause loop':'▶ Play loop';control.setAttribute('aria-pressed',String(playing));}
 function advance(t){if(playing&&visible&&!document.hidden&&slider.getClientRects().length){if(last){position+=(Math.min(t-last,100))/duration;if(position>=count())position=0;const n=Math.min(count()-1,Math.floor(position));if(n!==index())show(n);}paint();}last=t;requestAnimationFrame(advance);}
 slider.addEventListener('input',()=>{playing=false;position=Math.min(count()-1,Math.round(Number(slider.value)));show(position);paint();});
 control.onclick=()=>{playing=!playing;last=0;paint();};
 new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;last=0;},{threshold:.01}).observe(slider.closest('#projection-details')||$(section));
 reducedMotion.addEventListener('change',e=>{if(e.matches){playing=false;paint();}});
 paint();requestAnimationFrame(advance);
 return {reset(){position=0;show(0);paint();}};
}
stopFocus();stopTrace();showFocus(0);
makePlayer({section:'crowded',range:'focus-scrub',output:'focus-progress',button:'stage-play',count:()=>focus.length,index:()=>stage,show:showFocus});
const tracePlayer=makePlayer({section:'native-traces',range:'trace-scrub',output:'trace-progress',button:'trace-play',count:()=>traces[method].stages.length,index:()=>traceIndex,show:showTrace});
document.querySelectorAll('[data-method]').forEach(b=>b.onclick=()=>{selectMethod(b.dataset.method);tracePlayer.reset();});
// Decode in advance so playback never flashes unloaded stage assets.
[...new Set([...focus.flatMap(s=>[s[0],s[3]]),...Object.values(traces).flatMap(t=>t.stages.map(s=>s[0]))])].forEach(n=>{const im=new Image();im.src=asset(n);im.decode().catch(()=>{});});

$('copy-citation').addEventListener('click',async()=>{
 const content=$('citation-text').textContent;
 try{await navigator.clipboard.writeText(content);$('citation-status').textContent='BibTeX copied.';}
 catch{const selection=window.getSelection(),range=document.createRange();range.selectNodeContents($('citation-text'));selection.removeAllRanges();selection.addRange(range);$('citation-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';}
});

// A direct link to a folded demo opens it before scrolling into view.
function revealDemoAnchor(){
 let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}
 const target=document.getElementById(id);if(!target)return;
 let details=target.closest('details');while(details){details.open=true;details=details.parentElement.closest('details');}
 if(target.matches('details'))requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
}
window.addEventListener('hashchange',revealDemoAnchor);revealDemoAnchor();
