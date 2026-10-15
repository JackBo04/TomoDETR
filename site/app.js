'use strict';
const $ = id => document.getElementById(id);
let cases = [], current = 0, image = null, playing = false, last = 0, direction = 1, frame = 0, loadToken = 0;
const canvas = $('viewer'), ctx = canvas.getContext('2d');
const displayName = n => n === 'Local' ? 'w/o CWQR' : n;
function markers(points, type) {
  const colors = {tp:'#00e4b0',fp:'#ff5b72',fn:'#d99bff',gt:'#ffe361'};
  ctx.strokeStyle = colors[type]; ctx.lineWidth = 2.8;
  ctx.shadowColor = '#000'; ctx.shadowBlur = 3;
  ctx.setLineDash(type === 'fn' ? [5,4] : []);
  for (const [u,v] of points) {
    const x=u*800,y=v*800; ctx.beginPath();
    if(type==='fp'){ctx.moveTo(x-8,y-8);ctx.lineTo(x+8,y+8);ctx.moveTo(x-8,y+8);ctx.lineTo(x+8,y-8);}
    else ctx.arc(x,y,type==='gt'?13:10,0,Math.PI*2);
    ctx.stroke();
  }
  ctx.setLineDash([]);ctx.shadowBlur=0;
}
function draw() {
  if(!image || !cases.length) return;
  const c=cases[current],x=Number($('wipe').value)*8, method=$('baseline').value;
  ctx.clearRect(0,0,800,800);ctx.drawImage(image,0,0,800,800);
  for(const [name,left,right] of [[method,0,x],['TomoDETR',x,800]]) {
    ctx.save();ctx.beginPath();ctx.rect(left,0,right-left,800);ctx.clip();
    const m=c.methods[name];for(const kind of ['fn','tp','fp'])markers(m[kind],kind);
    if($('gt').checked)markers(c.gt,'gt');ctx.restore();
  }
  ctx.fillStyle='#112723dd';ctx.fillRect(22,738,c.scaleFraction*800+28,44);
  ctx.strokeStyle='white';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(36,754);ctx.lineTo(36+c.scaleFraction*800,754);ctx.stroke();ctx.fillStyle='white';ctx.font='15px Arial';ctx.fillText(`${c.scaleNm} nm`,36,775);
  $('divider').style.left=`${$('wipe').value}%`;
}
function updateStats(){
  if(!cases.length)return;
  const c=cases[current],method=$('baseline').value;
  $('left-label').textContent=displayName(method);$('count-baseline').textContent=displayName(method);
  ['tp','fp','fn'].forEach((k,i)=>{$(`base-${k}`).textContent=c.methods[method].counts[i];$(`ours-${k}`).textContent=c.methods.TomoDETR.counts[i];});
  canvas.setAttribute('aria-label',`${c.title}: ${displayName(method)} compared with TomoDETR. Correct / false positive / missed counts: ${c.methods[method].counts.join(' / ')} versus ${c.methods.TomoDETR.counts.join(' / ')}.`);
  draw();
}
async function selectCase(index){
  current=index;const c=cases[index],token=++loadToken;image=null;
  ctx.clearRect(0,0,800,800);
  $('dataset').textContent=`${c.dataset} · ${c.run}`;$('sample-title').textContent=c.title;$('sample-description').textContent=c.description;
  $('sample-note').textContent=`${c.gt.length} reference particles · ${c.slabNm.toFixed(1)} nm projection slab`;
  document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.case)===index)));
  if(index===1)$('baseline').value='Local';
  updateStats();const next=new Image();next.src=c.image;
  try{await next.decode();if(token!==loadToken)return;image=next;draw();}catch{$('load-error').hidden=false;$('load-error').textContent='The sample image could not be loaded. Please refresh to retry.';}
}
function stop(){playing=false;cancelAnimationFrame(frame);$('play').textContent='▶ Play comparison';$('play').setAttribute('aria-pressed','false');last=0;}
function animate(t){if(!playing)return;if(last){let v=Number($('wipe').value)+(t-last)*.018*direction;if(v>=95){v=95;direction=-1;}if(v<=5){v=5;direction=1;}$('wipe').value=String(v);draw();}last=t;frame=requestAnimationFrame(animate);}
$('play').addEventListener('click',()=>{if(playing){stop();return;}playing=true;last=0;$('play').textContent='Ⅱ Pause comparison';$('play').setAttribute('aria-pressed','true');frame=requestAnimationFrame(animate);});
$('wipe').addEventListener('input',()=>{stop();draw();});$('gt').addEventListener('change',draw);$('baseline').addEventListener('change',updateStats);
let dragging=false;const holder=$('canvas-wrap');
function drag(e){const b=holder.getBoundingClientRect();$('wipe').value=String(Math.max(0,Math.min(100,(e.clientX-b.left)/b.width*100)));draw();}
holder.addEventListener('pointerdown',e=>{stop();dragging=true;holder.setPointerCapture(e.pointerId);drag(e);});holder.addEventListener('pointermove',e=>{if(dragging)drag(e);});holder.addEventListener('pointerup',()=>{dragging=false;});holder.addEventListener('pointercancel',()=>{dragging=false;});
document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>selectCase(Number(b.dataset.case))));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText('pip install -e .');$('copy').textContent='Copied';}catch{$('copy').textContent='Select command to copy';}setTimeout(()=>{$('copy').textContent='Copy';},2400);});
fetch('assets/cases.json').then(r=>{if(!r.ok)throw new Error('Data request failed');return r.json();}).then(data=>{cases=data;return selectCase(0);}).catch(()=>{$('load-error').hidden=false;$('load-error').textContent='The comparison data could not be loaded. Please refresh to retry.';});
