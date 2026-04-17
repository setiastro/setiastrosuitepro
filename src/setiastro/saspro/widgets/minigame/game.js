/**
 * Neon Invaders: Hyperdrive
 * Core Game Engine
 */



'use strict';
// ─── CONSTANTS ──────────────────────────────────────────────────────────────
const GW=800, GH=600, FPS=60;
const INTERVAL=1000/FPS;

const C={
  PLAYER:'#0ff', PB:'#0ff', EB:'#f44',
  BASIC:'#f0f', FAST:'#3f3', HEAVY:'#f80',
  PULSAR:'#ff0', SPLITTER:'#0af', VOID_E:'#a0a',
  BOSS1:'#ddd', BOSS2:'#f0f', BOSS3:'#fa0', BOSS4:'#f44',
  PART:'#fa0', THRUST:'#09c'
};

// ─── INPUT ──────────────────────────────────────────────────────────────────
const Input={
  keys:{},
  init(){
    window.addEventListener('keydown',e=>{
      this.keys[e.key]=true;
      if(e.key===' ')e.preventDefault();
    });
    window.addEventListener('keyup',e=>{ this.keys[e.key]=false; });
  },
  is(k){ return !!this.keys[k]; }
};
Input.init();

// ─── UTILS ──────────────────────────────────────────────────────────────────
const rand=(a,b)=>Math.random()*(b-a)+a;
const rInt=(r1,r2)=>!(r2.left>r1.right||r2.right<r1.left||r2.top>r1.bottom||r2.bottom<r1.top);

// ─── AUDIO ──────────────────────────────────────────────────────────────────
const AudioSys={
  ac:null,
  init(){ try{ this.ac=new(window.AudioContext||window.webkitAudioContext)(); }catch(e){} },
  _tone(freqs,type,dur,vol=0.07){
    if(!this.ac)return;
    try{
      if(this.ac.state==='suspended')this.ac.resume();
      const o=this.ac.createOscillator(),g=this.ac.createGain();
      o.type=type;
      const t=this.ac.currentTime;
      if(Array.isArray(freqs)){ freqs.forEach((f,i)=>o.frequency.setValueAtTime(f,t+i*(dur/freqs.length))); }
      else o.frequency.setValueAtTime(freqs,t);
      g.gain.setValueAtTime(vol,t);
      g.gain.exponentialRampToValueAtTime(0.001,t+dur);
      o.connect(g); g.connect(this.ac.destination);
      o.start(); o.stop(t+dur);
    }catch(e){}
  },
  shoot(){ this._tone(520,'square',0.07,0.055); },
  enemyShoot(){ this._tone(160,'sawtooth',0.1,0.04); },
  hit(){ this._tone(200,'sawtooth',0.08,0.07); },
  explode(){ this._tone(80,'sawtooth',0.35,0.1); },
  powerup(){ this._tone([400,600,900],'sine',0.45,0.09); },
  combo(n){ this._tone(200+n*90,'square',0.14,0.07); },
  bossHit(){ this._tone([320,260,200],'sawtooth',0.22,0.09); },
  levelUp(){ this._tone([350,450,600,800],'sine',0.7,0.1); },
  seti(){ for(let i=0;i<8;i++) setTimeout(()=>this._tone(142+i*28,'sine',0.28,0.04),i*90); }
};
AudioSys.init();

// ─── TOASTS ─────────────────────────────────────────────────────────────────
const T={
  items:[],
  push(txt,frames=130,col='#0ff'){ this.items.push({text:String(txt),life:frames,max:frames,color:col}); },
  update(){ this.items=this.items.filter(t=>--t.life>0); },
  draw(ctx){
    if(!this.items.length)return;
    const shown=this.items.slice(-3);
    ctx.save();
    ctx.font='bold 15px Orbitron,monospace';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    shown.forEach((t,i)=>{
      const a=Math.min(1,t.life/40);
      const y=88+i*32;
      const w=Math.min(GW-60,ctx.measureText(t.text).width+56);
      ctx.globalAlpha=0.88*a;
      ctx.fillStyle='rgba(0,0,0,0.65)';
      ctx.strokeStyle=t.color; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.rect(GW/2-w/2,y-14,w,28); ctx.fill(); ctx.stroke();
      ctx.globalAlpha=a; ctx.fillStyle=t.color;
      ctx.fillText(t.text,GW/2,y);
    });
    ctx.restore(); ctx.globalAlpha=1;
  }
};

// ─── SECRET CODES ───────────────────────────────────────────────────────────
const Secrets={
  buf:[], last:0,
  feed(c){
    const now=performance.now();
    if(now-this.last>1600) this.buf=[];
    this.last=now;
    this.buf.push(c.toUpperCase());
    if(this.buf.length>14) this.buf.shift();
    const s=this.buf.join('');
    if(s.endsWith('SOLVE'))  { this.buf=[]; this._solve(); }
    if(s.endsWith('DARKS'))  { this.buf=[]; this._darks(); }
    if(s.endsWith('FLATS'))  { this.buf=[]; this._flats(); }
    if(s.endsWith('DRIZZLE')){ this.buf=[]; this._drizzle(); }
    if(s.endsWith('SETI'))   { this.buf=[]; this._seti(); }
  },
  _solve(){
    if(gState!=='PLAYING'||!G.player||G.player.dead)return;
    T.push('ASTAP SOLVED  0.92"/px  WCS LOCKED',190,'#0ff');
    T.push('+1 SHIELD  +500 PTS',140,'#0f0');
    G.player.shield=Math.min(3,(G.player.shield||0)+1);
    score+=500; updHUD();
  },
  _darks(){
    if(gState!=='PLAYING')return;
    T.push('DARK SUBTRACTION — ENEMIES HIDDEN!',190,'#888');
    G.darksMode=150;
  },
  _flats(){
    if(gState!=='PLAYING')return;
    T.push('FLAT CALIBRATION — SCREEN WASH!',130,'#fff');
    G.flatsFlash=35;
  },
  _drizzle(){
    if(gState!=='PLAYING'||!G.player)return;
    T.push('DRIZZLE MODE — SUB-PIXEL SPREAD!',190,'#0af');
    G.player.drizzle=320;
  },
  _seti(){
    if(gState!=='PLAYING')return;
    T.push('SETI ARRAY ONLINE — 1420 MHz',190,'#0ff');
    G.setiMode=380; G.setiX=-60; AudioSys.seti();
  }
};
window.addEventListener('keydown',e=>{
  if(e.key&&e.key.length===1&&/[a-zA-Z]/.test(e.key)) Secrets.feed(e.key);
});

// ─── LEADERBOARD (Supabase) ───────────────────────────────────────────────────
const SB_URL = 'https://rtdsyiudcmfznyccbkta.supabase.co';
const SB_KEY = 'sb_publishable_5Ajgy3eW-ddgNBFfXujYpw_2V23vx8Y';

function lbSave(name, sc, lv) {
    // Save to localStorage as backup
    const rows = JSON.parse(localStorage.getItem('ni_lb_v2') || '[]');
    rows.push({ name: name.toUpperCase().slice(0, 12), score: sc, level: lv, ts: Date.now() });
    rows.sort((a, b) => b.score - a.score);
    localStorage.setItem('ni_lb_v2', JSON.stringify(rows.slice(0, 50)));

    // Save to Supabase
    fetch(`${SB_URL}/rest/v1/leaderboard`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SB_KEY,
            'Authorization': `Bearer ${SB_KEY}`
        },
        body: JSON.stringify({
            name: name.toUpperCase().slice(0, 12),
            score: sc,
            level: lv
        })
    }).catch(() => {});
}

function lbGet() {
    return JSON.parse(localStorage.getItem('ni_lb_v2') || '[]')
        .sort((a, b) => b.score - a.score).slice(0, 20);
}

function lbRender() {
    const tbody = document.getElementById('lb-body');
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#fa0;padding:20px">LOADING...</td></tr>';

    fetch(`${SB_URL}/rest/v1/leaderboard?select=name,score,level&order=score.desc&limit=20`, {
        headers: {
            'apikey': SB_KEY,
            'Authorization': `Bearer ${SB_KEY}`
        }
    })
    .then(r => r.json())
    .then(rows => {
        if (!rows || !rows.length) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#444;padding:20px">NO SCORES YET — BE FIRST!</td></tr>';
            return;
        }
        tbody.innerHTML = rows.map((r, i) =>
            `<tr>
                <td style="color:#fa0">${i + 1}</td>
                <td style="color:#0ff">${r.name}</td>
                <td style="color:#0f0">${Number(r.score).toLocaleString()}</td>
                <td style="color:#f0f">L${r.level}</td>
            </tr>`
        ).join('');
    })
    .catch(() => {
        const rows = lbGet();
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#444;padding:20px">NO SCORES YET — BE FIRST!</td></tr>';
            return;
        }
        tbody.innerHTML = rows.map((r, i) =>
            `<tr>
                <td style="color:#fa0">${i + 1}</td>
                <td style="color:#0ff">${r.name}</td>
                <td style="color:#0f0">${r.score.toLocaleString()}</td>
                <td style="color:#f0f">L${r.level}</td>
            </tr>`
        ).join('');
        T.push('OFFLINE — SHOWING LOCAL SCORES', 120, '#fa0');
    });
}
// ─── CANVAS + GLOBALS ────────────────────────────────────────────────────────
const canvas=document.getElementById('gameCanvas');
const ctx=canvas.getContext('2d');
canvas.width=GW; canvas.height=GH;

let gState='MENU', score=0, level=1, frameCount=0;
let combo=0, comboTimer=0, lastKillTime=0;

// ─── PARTICLE ───────────────────────────────────────────────────────────────
class Particle{
  constructor(x,y,col,spd,life){
    this.x=x; this.y=y; this.color=col; this.life=life; this.max=life;
    const a=rand(0,Math.PI*2);
    this.vx=Math.cos(a)*spd; this.vy=Math.sin(a)*spd;
    this.s=rand(1,3.2);
  }
  update(){ this.x+=this.vx; this.y+=this.vy; this.life--; this.s*=0.93; }
  draw(ctx){
    ctx.globalAlpha=this.life/this.max;
    ctx.fillStyle=this.color;
    ctx.beginPath(); ctx.arc(this.x,this.y,Math.max(0.1,this.s),0,Math.PI*2); ctx.fill();
    ctx.globalAlpha=1;
  }
}

// ─── STARFIELD ──────────────────────────────────────────────────────────────
class Starfield{
  constructor(){ this.stars=[]; for(let i=0;i<130;i++) this._add(true); }
  _add(randY=false){
    this.stars.push({x:rand(0,GW),y:randY?rand(0,GH):0,z:rand(0.3,2.8),s:rand(0.4,2.1)});
  }
  update(){
    this.stars.forEach(s=>{
      s.y+=1.1*s.z;
      if(s.y>GH){ s.y=0; s.x=rand(0,GW); }
    });
  }
  draw(ctx){
    const palettes=[
      ['#ddd','#0ff'],['#ddd','#0f0'],['#ddd','#ff0'],
      ['#ddd','#f0f'],['#ddd','#f44'],['#adf','#0ff'],
      ['#dfd','#0f0'],['#ffd','#ff0'],['#fdf','#f0f'],['#fdd','#f44']
    ];
    const pal=palettes[(level-1)%palettes.length];
    this.stars.forEach(s=>{
      ctx.globalAlpha=s.z/3.5;
      ctx.fillStyle=s.z>1.8?pal[1]:pal[0];
      ctx.beginPath(); ctx.arc(s.x,s.y,s.s,0,Math.PI*2); ctx.fill();
    });
    ctx.globalAlpha=1;
  }
}

// ─── BULLET ─────────────────────────────────────────────────────────────────
class Bullet{
  constructor(x,y,vy,isPlayer,vx=0,big=false){
    this.x=x; this.y=y; this.vx=vx; this.vy=vy;
    this.isPlayer=isPlayer; this.active=true; this.big=big;
    this.w=big?9:4; this.h=big?15:11;
  }
  update(){
    this.x+=this.vx; this.y+=this.vy;
    if(this.y<-70||this.y>GH+70||this.x<-70||this.x>GW+70) this.active=false;
  }
  draw(ctx){
    const col=this.isPlayer?C.PB:C.EB;
    ctx.fillStyle=col; ctx.shadowBlur=this.big?14:5; ctx.shadowColor=col;
    if(this.big){
      ctx.beginPath();
      ctx.ellipse(this.x,this.y,this.w/2,this.h/2,0,0,Math.PI*2);
      ctx.fill();
    } else {
      ctx.fillRect(this.x-this.w/2,this.y-this.h/2,this.w,this.h);
    }
    ctx.shadowBlur=0;
  }
  getBounds(){ return {left:this.x-this.w/2,right:this.x+this.w/2,top:this.y-this.h/2,bottom:this.y+this.h/2}; }
}

// ─── NOVA RING ───────────────────────────────────────────────────────────────
class NovaRing{
  constructor(x,y){
    this.x=x; this.y=y; this.r=0; this.maxR=520;
    this.life=1.0; this.active=true;
  }
  update(){
    this.r+=18; this.life=1-(this.r/this.maxR);
    if(this.r>=this.maxR) this.active=false;
  }
  draw(ctx){
    const a=this.life;
    // outer ring
    ctx.save();
    ctx.globalAlpha=a*0.9;
    ctx.strokeStyle=`rgba(255,170,0,${a})`;
    ctx.lineWidth=6;
    ctx.shadowColor='#fa0'; ctx.shadowBlur=30;
    ctx.beginPath(); ctx.arc(this.x,this.y,this.r,0,Math.PI*2); ctx.stroke();
    // inner glow ring
    ctx.strokeStyle=`rgba(255,255,255,${a*0.6})`;
    ctx.lineWidth=2; ctx.shadowBlur=10;
    ctx.beginPath(); ctx.arc(this.x,this.y,this.r*0.85,0,Math.PI*2); ctx.stroke();
    // fill flash near center
    if(this.r<80){
      ctx.globalAlpha=a*(1-this.r/80)*0.4;
      ctx.fillStyle='#fff';
      ctx.beginPath(); ctx.arc(this.x,this.y,this.r,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();
  }
}

// ─── POWERUP ────────────────────────────────────────────────────────────────
class Powerup{
  constructor(type,x){
    this.type=type;
    this.x=x||rand(40,GW-40); this.y=-30;
    this.speed=1.75; this.active=true; this.t=0;
    const cols={hp:'#f44',fire:'#0af',shield:'#0ff',bomb:'#fa0'};
    this.col=cols[type]||'#f0f';
  }
  update(){ this.y+=this.speed; this.t++; if(this.y>GH+35) this.active=false; }
  draw(ctx){
    ctx.save();
    const pulse=1+Math.sin(this.t*.16)*.1;
    ctx.translate(this.x,this.y); ctx.scale(pulse,pulse);
    ctx.shadowBlur=16; ctx.shadowColor=this.col;
    ctx.strokeStyle=this.col; ctx.fillStyle=this.col; ctx.lineWidth=2;
    ctx.beginPath();
    if(this.type==='hp'){
      ctx.fillRect(-10,-3,20,6); ctx.fillRect(-3,-10,6,20);
    } else if(this.type==='fire'){
      ctx.moveTo(0,-15); ctx.lineTo(10,0); ctx.lineTo(-4,0);
      ctx.lineTo(4,15); ctx.lineTo(-10,0); ctx.lineTo(4,0); ctx.closePath(); ctx.fill();
    } else if(this.type==='shield'){
      ctx.arc(0,0,12,0,Math.PI*2); ctx.stroke();
      ctx.beginPath(); ctx.arc(0,0,6,0,Math.PI*2); ctx.fill();
    } else if(this.type==='bomb'){
      ctx.arc(0,0,11,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#000'; ctx.beginPath(); ctx.arc(0,0,5,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();
  }
  getBounds(){ return {left:this.x-15,right:this.x+15,top:this.y-15,bottom:this.y+15}; }
}

// ─── PLAYER ─────────────────────────────────────────────────────────────────
class Player{
  constructor(){
    this.x=GW/2; this.y=GH-90;
    this.spd=5.5; this.hp=100; this.maxHp=100;
    this.cooldown=0; this.fireRate=13; this.dead=false;
    this.shield=0; this.drizzle=0; this.iframes=0;
    this.fireTimer=null; this.trail=[];
  }
  update(){
    if(this.dead)return;
    let dx=0,dy=0;
    if(Input.is('ArrowLeft')) dx-=this.spd;
    if(Input.is('ArrowRight')) dx+=this.spd;
    if(Input.is('ArrowUp')) dy-=this.spd;
    if(Input.is('ArrowDown')) dy+=this.spd;
    if(dx&&dy){ dx*=0.707; dy*=0.707; }
    this.x=Math.max(18,Math.min(GW-18,this.x+dx));
    this.y=Math.max(18,Math.min(GH-18,this.y+dy));
    this.trail.unshift({x:this.x,y:this.y});
    if(this.trail.length>14) this.trail.pop();
    if(this.cooldown>0) this.cooldown--;
    if(this.iframes>0) this.iframes--;
    if(this.drizzle>0) this.drizzle--;
    // Nova bomb
    if(Input.is('Enter')&&!this._bombHeld&&G.bombStock>0){
      this._bombHeld=true;
      G._fireBomb();
    }
    if(!Input.is('Enter')) this._bombHeld=false;    
    if((Input.is(' '))&&this.cooldown<=0){ this._fire(); this.cooldown=this.fireRate; }
    if(Math.random()<0.55)
      G.particles.push(new Particle(this.x+rand(-6,6),this.y+22,C.THRUST,1.3,14));
  }
  _fire(){
    const spd=-12.5;
    if(this.drizzle>0){
      for(let i=-3;i<=3;i++){
        const vx=i*0.85+rand(-.3,.3);
        G.bullets.push(new Bullet(this.x+vx*4,this.y-14,spd+rand(-.4,.4),true,vx));
      }
    } else if(this.fireRate<=7){
      G.bullets.push(new Bullet(this.x-9,this.y-10,spd,true));
      G.bullets.push(new Bullet(this.x+9,this.y-10,spd,true));
    } else {
      G.bullets.push(new Bullet(this.x,this.y-20,spd,true));
    }
    AudioSys.shoot();
  }
  draw(ctx){
    if(this.dead)return;
    // trail
    this.trail.forEach((p,i)=>{
      ctx.globalAlpha=(1-i/14)*0.22;
      ctx.fillStyle=C.PLAYER;
      ctx.beginPath(); ctx.arc(p.x,p.y,3*(1-i/14),0,Math.PI*2); ctx.fill();
    });
    ctx.globalAlpha=1;
    if(this.iframes>0&&Math.floor(this.iframes/4)%2===0) return;
    ctx.strokeStyle=C.PLAYER; ctx.lineWidth=2.5;
    ctx.shadowColor=C.PLAYER; ctx.shadowBlur=14;
    ctx.beginPath();
    ctx.moveTo(this.x,this.y-22);
    ctx.lineTo(this.x+19,this.y+18);
    ctx.lineTo(this.x,this.y+8);
    ctx.lineTo(this.x-19,this.y+18);
    ctx.closePath(); ctx.stroke();
    ctx.fillStyle='rgba(0,255,255,0.12)'; ctx.fill();
    // engine glow
    ctx.beginPath(); ctx.arc(this.x,this.y+10,4,0,Math.PI*2);
    ctx.fillStyle='rgba(0,180,255,0.6)'; ctx.fill();
    // shield ring
    if(this.shield>0){
      const pulse=0.3+0.25*Math.sin(frameCount*.12);
      ctx.beginPath(); ctx.arc(this.x,this.y,30,0,Math.PI*2);
      ctx.strokeStyle=`rgba(0,255,255,${pulse})`; ctx.lineWidth=3; ctx.stroke();
    }
    ctx.shadowBlur=0;
  }
  hit(dmg){
    if(this.iframes>0)return;
    if(this.shield>0){
      this.shield--; AudioSys.hit();
      T.push('SHIELD ABSORBED',65,'#0ff');
      updHUD(); return;
    }
    this.hp-=dmg; this.iframes=45;
    AudioSys.hit(); updHUD(); G.triggerShake(7);
    if(this.hp<=0){
      this.hp=0; this.dead=true;
      G.explode(this.x,this.y,65,C.PLAYER);
      setTimeout(()=>G.gameOver(),1300);
    }
  }
  getBounds(){ return {left:this.x-13,right:this.x+13,top:this.y-14,bottom:this.y+14}; }
}

// ─── ENEMY ──────────────────────────────────────────────────────────────────
class Enemy{
  constructor(type,overrides={}){
    this.type=type; this.active=true;
    this.timer=Math.floor(rand(0,110));
    this.hitFlash=0;
    const spd=1+level*0.08;
    switch(type){
      case'basic':
        this.hp=this.maxHp=2; this.score=100; this.spd=1.3+spd*.4;
        this.col=C.BASIC; this.w=28; this.h=28; this.shootInt=190; break;
      case'fast':
        this.hp=this.maxHp=1; this.score=65; this.spd=3.2+spd*.5;
        this.col=C.FAST; this.w=20; this.h=20; this.shootInt=0; break;
      case'heavy':
        this.hp=this.maxHp=7+level; this.score=300; this.spd=0.65+spd*.2;
        this.col=C.HEAVY; this.w=50; this.h=50; this.shootInt=110; break;
      case'pulsar':
        this.hp=this.maxHp=4; this.score=220; this.spd=0.35;
        this.col=C.PULSAR; this.w=34; this.h=34; this.shootInt=85;
        this.pulseAngle=0; break;
      case'splitter':
        this.hp=this.maxHp=3; this.score=190; this.spd=1.7+spd*.35;
        this.col=C.SPLITTER; this.w=32; this.h=32; this.shootInt=220;
        this.isChild=false; break;
    }
    Object.assign(this,overrides);
    this.x=rand(40,GW-40); this.y=-65;
  }
  update(){
    this.y+=this.spd; this.timer++;
    this.hitFlash=Math.max(0,this.hitFlash-1);
    if(this.type==='fast') this.x+=Math.sin(this.timer*.16)*2.8;
    if(this.type==='pulsar'){
      this.pulseAngle+=0.055;
      if(this.y>90&&this.y<GH-90&&this.timer%this.shootInt===0){
        for(let i=0;i<8;i++){
          const a=i*Math.PI/4+this.pulseAngle;
          G.bullets.push(new Bullet(this.x,this.y,Math.sin(a)*3.8,false,Math.cos(a)*3.8));
        }
        AudioSys.enemyShoot();
      }
    } else if(this.shootInt>0&&this.timer%this.shootInt===0&&this.y>70&&this.y<GH-70){
      if(G.player&&!G.player.dead){
        const dx=G.player.x-this.x, dy=G.player.y-this.y;
        const len=Math.sqrt(dx*dx+dy*dy)||1;
        G.bullets.push(new Bullet(this.x,this.y,dy/len*4.8,false,dx/len*4.8));
        AudioSys.enemyShoot();
      }
    }
    this.x=Math.max(20,Math.min(GW-20,this.x));
  }
  draw(ctx){
    const col=this.hitFlash>0?'#fff':this.col;
    ctx.strokeStyle=col; ctx.lineWidth=2.2;
    ctx.shadowColor=col; ctx.shadowBlur=10;
    ctx.fillStyle=this.hitFlash>0?'rgba(255,255,255,0.3)':'rgba(0,0,0,0)';
    ctx.beginPath();
    if(this.type==='basic'){
      ctx.moveTo(this.x,this.y-14); ctx.lineTo(this.x+14,this.y+8);
      ctx.lineTo(this.x+8,this.y+14); ctx.lineTo(this.x-8,this.y+14);
      ctx.lineTo(this.x-14,this.y+8); ctx.closePath();
    } else if(this.type==='fast'){
      ctx.moveTo(this.x,this.y+11); ctx.lineTo(this.x+10,this.y-10);
      ctx.lineTo(this.x-10,this.y-10); ctx.closePath();
    } else if(this.type==='heavy'){
      ctx.arc(this.x,this.y,23,0,Math.PI*2);
    } else if(this.type==='pulsar'){
      const r=14+5*Math.sin(this.pulseAngle*2);
      ctx.arc(this.x,this.y,r,0,Math.PI*2);
      ctx.stroke(); ctx.beginPath();
      ctx.strokeStyle=col.replace(')',',0.25)').replace('rgb','rgba');
      ctx.lineWidth=10; ctx.arc(this.x,this.y,r+6,0,Math.PI*2);
      ctx.lineWidth=2.2; ctx.strokeStyle=col;
    } else if(this.type==='splitter'){
      ctx.rect(this.x-14,this.y-14,28,28);
    }
    ctx.stroke(); ctx.fill();
    // HP bar for tanky enemies
    if((this.type==='heavy'||this.type==='pulsar')&&this.hp<this.maxHp){
      const pct=this.hp/this.maxHp;
      ctx.fillStyle='#222'; ctx.fillRect(this.x-18,this.y-this.h/2-10,36,4);
      ctx.fillStyle=pct>0.5?'#0f0':pct>0.25?'#fa0':'#f44';
      ctx.fillRect(this.x-18,this.y-this.h/2-10,36*pct,4);
    }
    ctx.shadowBlur=0;
  }
  hit(dmg){
    if(this.y<0)return;
    this.hp-=dmg; this.hitFlash=7; AudioSys.hit();
    if(this.hp<=0){
      this.active=false; score+=this.score;
      if(this.type==='splitter'&&!this.isChild){
        for(let i=0;i<2;i++){
          const e=new Enemy('fast',{isChild:true,score:30});
          e.x=this.x+(i===0?-22:22); e.y=this.y; e.spd=4;
          G.enemies.push(e);
        }
      }
      G.explode(this.x,this.y,24,this.col); updHUD();
      // combo
      const now=performance.now(); const dt=now-lastKillTime; lastKillTime=now;
      if(dt<2600){ combo++; comboTimer=130; AudioSys.combo(Math.min(combo,10));
        if(combo>=2){
          score+=combo*55;
          const cd=document.getElementById('combo-disp');
          if(cd){ cd.textContent=`x${combo} COMBO!`; cd.style.opacity=1; }
        }
      } else { combo=1; comboTimer=90; }
      G.checkProgress();
    }
  }
  getBounds(){
    const hw=this.w/2,hh=this.h/2;
    return {left:this.x-hw,right:this.x+hw,top:this.y-hh,bottom:this.y+hh};
  }
}

// ─── BOSS ───────────────────────────────────────────────────────────────────
class Boss{
  constructor(tier){
    this.tier=tier||1; this.x=GW/2; this.y=-130;
    this.active=true; this.phase=1; this.dir=1;
    this.timer=0; this.iframes=0; this.hitFlash=0;
    this.shieldGens=[]; this.shielded=false; this.rageMode=false;
    this.w=130; this.h=90;
    switch(tier){
      case 1: this.hp=this.maxHp=700; this.col=C.BOSS1; this.name='VOID KEEPER'; break;
      case 2: this.hp=this.maxHp=1000; this.col=C.BOSS2; this.name='NEBULA HORROR'; break;
      case 3: this.hp=this.maxHp=1400; this.col=C.BOSS3; this.name='PULSAR TYRANT'; break;
      case 4: this.hp=this.maxHp=2000; this.col=C.BOSS4; this.name='DARK SINGULARITY'; break;
    }
    this.score=3500*tier;
    document.getElementById('boss-bar-wrap').style.display='block';
    document.getElementById('boss-name').textContent=this.name;
    document.getElementById('boss-bar-fill').style.background=this.col;
    T.push(`⚠ WARNING: ${this.name} INCOMING ⚠`,220,'#f00');
    AudioSys.bossHit();
  }
  update(){
    this.timer++; this.hitFlash=Math.max(0,this.hitFlash-1);
    if(this.iframes>0) this.iframes--;
    const pct=this.hp/this.maxHp;
    // phase transitions
    if(pct<=0.25&&this.phase<4){
      this.phase=4; this.rageMode=true;
      T.push('RAGE MODE ACTIVATED',160,'#f44');
    } else if(pct<=0.5&&this.phase<3){
      this.phase=3;
      T.push('PHASE 3 — PATTERN CHANGE',130,'#fa0');
    } else if(pct<=0.75&&this.phase<2){
      this.phase=2;
      this.shieldGens=[];
      for(let i=-1;i<=1;i+=2){
        this.shieldGens.push({x:this.x+i*80,y:this.y+20,hp:3,active:true,col:'#0ff'});
      }
      this.shielded=true;
      T.push('SHIELD GENERATORS ONLINE!',160,'#0ff');
    }
    // movement
    if(this.y<105) this.y+=1.4;
    else{
      const spd=(this.rageMode?3.8:2.1)+(this.phase*.3);
      this.x+=spd*this.dir;
      if(this.x>GW-90||this.x<90) this.dir*=-1;
    }
    // sync shield gen positions
    this.shieldGens.forEach((s,i)=>{
      s.x+=((this.rageMode?3.8:2.1)+(this.phase*.3))*this.dir;
      s.x=Math.max(35,Math.min(GW-35,s.x));
      s.y=this.y+15;
    });
    this.shieldGens=this.shieldGens.filter(s=>s.active);
    if(this.shieldGens.length===0&&this.shielded){
      this.shielded=false; T.push('SHIELDS DOWN!',110,'#f00');
    }
    if(this.y>=105){
      if(this.timer%52===0)  this._tripleShot();
      if(this.timer%155===0) this._spreadShot();
      if(this.phase>=2&&this.timer%3===0) this._spiral();
      if(this.phase>=4&&this.timer%72===0) this._crossShot();
      if(this.tier>=3&&this.timer%190===0) this._pulsarRing();
    }
    // update bar
    const fill=document.getElementById('boss-bar-fill');
    if(fill) fill.style.width=Math.max(0,this.hp/this.maxHp*100)+'%';
  }
  _tripleShot(){
    [-2,0,2].forEach(vx=>G.bullets.push(new Bullet(this.x+vx*18,this.y+48,5.5,false,vx)));
    AudioSys.enemyShoot();
  }
  _spreadShot(){
    for(let i=-5;i<=5;i++)
      G.bullets.push(new Bullet(this.x,this.y+48,4.8,false,i*1.6));
  }
  _spiral(){
    const rate=this.rageMode?0.18:0.12;
    const ang=this.timer*rate;
    const spd=this.rageMode?3.8:2.8;
    G.bullets.push(new Bullet(this.x,this.y+38,Math.sin(ang)*spd+2.8,false,Math.cos(ang)*spd));
  }
  _crossShot(){
    const dirs=[[1,0],[-1,0],[0,1],[0,-1],[.7,.7],[-.7,.7],[.7,-.7],[-.7,-.7]];
    dirs.forEach(([dx,dy])=>G.bullets.push(new Bullet(this.x,this.y+38,dy*5.5,false,dx*5.5)));
  }
  _pulsarRing(){
    for(let i=0;i<12;i++){
      const a=i*(Math.PI/6);
      G.bullets.push(new Bullet(this.x,this.y+38,Math.sin(a)*4.2,false,Math.cos(a)*4.2));
    }
  }
  draw(ctx){
    const col=this.hitFlash>0?'#fff':this.col;
    const rage=this.rageMode;
    ctx.shadowColor=col; ctx.shadowBlur=rage?35:22;
    ctx.strokeStyle=col; ctx.lineWidth=rage?4:2.8;
    ctx.fillStyle=rage?'rgba(255,0,0,0.1)':'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.moveTo(this.x,this.y+48); ctx.lineTo(this.x+68,this.y-22);
    ctx.lineTo(this.x+38,this.y-48); ctx.lineTo(this.x,this.y-32);
    ctx.lineTo(this.x-38,this.y-48); ctx.lineTo(this.x-68,this.y-22);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    // reactor core
    ctx.fillStyle=rage?'#f00':'#f44';
    ctx.shadowColor='#f44'; ctx.shadowBlur=18;
    ctx.beginPath(); ctx.arc(this.x,this.y,rage?22:15,0,Math.PI*2); ctx.fill();
    // shield generators
    this.shieldGens.forEach(s=>{
      ctx.strokeStyle='#0ff'; ctx.lineWidth=2; ctx.shadowColor='#0ff'; ctx.shadowBlur=14;
      ctx.beginPath(); ctx.arc(s.x,s.y,15,0,Math.PI*2); ctx.stroke();
      ctx.strokeStyle='rgba(0,255,255,0.5)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.arc(s.x,s.y,22,0,Math.PI*2); ctx.stroke();
    });
    if(this.shielded){
      ctx.strokeStyle=`rgba(0,255,255,${0.15+0.1*Math.sin(frameCount*.1)})`;
      ctx.lineWidth=8;
      ctx.beginPath(); ctx.arc(this.x,this.y,90,0,Math.PI*2); ctx.stroke();
    }
    ctx.shadowBlur=0;
  }
  hitShieldGen(b){
    let hit=false;
    this.shieldGens.forEach(s=>{
      if(!s.active)return;
      const sb={left:s.x-17,right:s.x+17,top:s.y-17,bottom:s.y+17};
      if(rInt(b.getBounds(),sb)){
        s.hp--; hit=true;
        G.particles.push(new Particle(s.x,s.y,'#0ff',3,20));
        if(s.hp<=0){ s.active=false; G.explode(s.x,s.y,22,'#0ff'); }
      }
    });
    return hit;
  }
  hit(dmg){
    if(this.iframes>0)return false;
    if(this.shielded){
      T.push('SHIELD GENERATORS MUST DIE FIRST',80,'#0ff');
      return false;
    }
    this.hp-=(this.rageMode?dmg*2:dmg);
    this.hitFlash=5; this.iframes=2; AudioSys.bossHit();
    const fill=document.getElementById('boss-bar-fill');
    if(fill) fill.style.width=Math.max(0,this.hp/this.maxHp*100)+'%';
    if(this.hp<=0){
      this.active=false; score+=this.score;
      G.explode(this.x,this.y,130,'#fff');
      document.getElementById('boss-bar-wrap').style.display='none';
      updHUD(); G.victory();
    }
    return true;
  }
  getBounds(){ return {left:this.x-58,right:this.x+58,top:this.y-38,bottom:this.y+38}; }
}

// ─── GAME MANAGER ────────────────────────────────────────────────────────────
const G={
  player:null, bullets:[], enemies:[], particles:[], powerups:[], boss:null,
  novaRings:[], bombStock:3,
  starfield:null, waveTimer:0, enemiesToSpawn:0, spawnTimer:0, spawnInterval:80,
  powerupQueue:[], shake:0,
  darksMode:0, flatsFlash:0, setiMode:0, setiX:-60,

  triggerShake(a){ this.shake=Math.max(this.shake,a); },
  explode(x,y,n,col){
    AudioSys.explode(); this.triggerShake(n/4);
    for(let i=0;i<n;i++) this.particles.push(new Particle(x,y,col,rand(1.2,5.5),rand(16,50)));
  },

  _fireBomb(){
    if(this.bombStock<=0)return;
    this.bombStock--;
    updHUD();
    this.novaRings.push(new NovaRing(GW/2,GH/2));
    this.triggerShake(20);
    T.push(`NOVA BOMB — ${this.bombStock} REMAINING`,130,'#fa0');
    AudioSys.explode();
    // Deal 5 hits to everything over 5 frames
    let hits=0;
    const interval=setInterval(()=>{
      this.enemies.forEach(e=>e.hit(1));
      if(this.boss&&this.boss.active) this.boss.hit(1);
      hits++;
      if(hits>=5) clearInterval(interval);
    },80);
  },

  init(){
    this.starfield=new Starfield();
    this._setupUI();
    requestAnimationFrame(loop);
  },

  _setupUI(){
    document.getElementById('btn-start').onclick=()=>this.start(1);
    document.getElementById('btn-levels').onclick=()=>{
      document.getElementById('main-menu').classList.add('hidden');
      document.getElementById('level-select').classList.remove('hidden');
      this._renderLevelGrid();
    };
    document.getElementById('btn-lb').onclick=()=>{
      document.getElementById('main-menu').classList.add('hidden');
      document.getElementById('leaderboard').classList.remove('hidden');
      lbRender();
    };
    document.getElementById('btn-back').onclick=()=>{
      document.getElementById('level-select').classList.add('hidden');
      document.getElementById('main-menu').classList.remove('hidden');
    };
    document.getElementById('btn-lb-back').onclick=()=>{
      document.getElementById('leaderboard').classList.add('hidden');
      document.getElementById('main-menu').classList.remove('hidden');
    };
    document.getElementById('btn-retry').onclick=()=>this.start(level);
    document.getElementById('btn-menu').onclick=()=>this.showMenu();
    document.getElementById('btn-menu-win').onclick=()=>this.showMenu();
    document.getElementById('btn-next-level').onclick=()=>this.start(Math.min(level+1,10));
    document.getElementById('btn-submit-score').onclick=()=>this._doSubmit('player-name','btn-submit-score');
    document.getElementById('btn-submit-vscore').onclick=()=>this._doSubmit('vplayer-name','btn-submit-vscore');
    // Stop keydown from controlling ship when typing name
    ['player-name','vplayer-name'].forEach(id=>{
      const el=document.getElementById(id);
      if(el) el.addEventListener('keydown',e=>e.stopPropagation());
    });
  },

  _doSubmit(inputId,btnId){
    const n=document.getElementById(inputId).value.trim()||'UNKNOWN';
    lbSave(n,score,level);
    T.push('SCORE SAVED TO LEADERBOARD!',150,'#0f0');
    const btn=document.getElementById(btnId);
    btn.textContent='SAVED!'; btn.disabled=true;
  },

  _renderLevelGrid(){
    const g=document.getElementById('level-grid'); g.innerHTML='';
    for(let i=1;i<=10;i++){
      const b=document.createElement('button'); b.className='lvl-btn';
      b.textContent=`LEVEL ${i}`; b.onclick=()=>this.start(i);
      g.appendChild(b);
    }
  },

  start(lv){
    level=lv; score=0; combo=0; comboTimer=0; lastKillTime=0; frameCount=0;
    this.player=new Player();
    this.bullets=[]; this.enemies=[]; this.particles=[];
    this.powerups=[]; this.boss=null;
    this.waveTimer=0; this.spawnTimer=0;
    this.darksMode=0; this.flatsFlash=0; this.setiMode=0;
    this.bombStock=3; this.novaRings=[];    
    this.enemiesToSpawn=14+level*9;
    this.spawnInterval=Math.max(42,105-level*6);
    this._schedulePowerups();
    document.getElementById('boss-bar-wrap').style.display='none';
    document.getElementById('boss-bar-fill').style.width='100%';
    ['btn-submit-score','btn-submit-vscore'].forEach(id=>{
      const b=document.getElementById(id); b.disabled=false; b.textContent='SUBMIT SCORE';
    });
    ['player-name','vplayer-name'].forEach(id=>{
      const el=document.getElementById(id); if(el) el.value='';
    });
    ['main-menu','level-select','game-over','victory','leaderboard'].forEach(id=>
      document.getElementById(id).classList.add('hidden'));
    document.getElementById('hud').classList.remove('hidden');
    const cd=document.getElementById('combo-disp'); if(cd) cd.style.opacity=0;
    gState='PLAYING'; updHUD();
    T.push(`LEVEL ${level} — ENGAGE`,130,'#0ff');
  },

  _schedulePowerups(){
    this.powerupQueue=[]; let t=rand(220,420);
    this.powerupQueue.push({type:'fire',time:t});
    const hpC=1+Math.floor(level/3);
    for(let i=0;i<hpC;i++){ t+=rand(190,430); this.powerupQueue.push({type:'hp',time:t}); }
    if(level>=2){ t+=rand(220,450); this.powerupQueue.push({type:'shield',time:t}); }
    if(level>=4){ t+=rand(280,500); this.powerupQueue.push({type:'bomb',time:t}); }
    if(level>=6){ t+=rand(300,550); this.powerupQueue.push({type:'shield',time:t}); }
  },

  showMenu(){
    gState='MENU';
    ['game-over','victory','hud'].forEach(id=>document.getElementById(id).classList.add('hidden'));
    document.getElementById('main-menu').classList.remove('hidden');
    document.getElementById('boss-bar-wrap').style.display='none';
  },

  gameOver(){
    gState='GAMEOVER';
    document.getElementById('final-score').textContent=score.toLocaleString();
    document.getElementById('hud').classList.add('hidden');
    document.getElementById('game-over').classList.remove('hidden');
  },

  victory(){
    gState='VICTORY';
    document.getElementById('victory-score').textContent=score.toLocaleString();
    document.getElementById('hud').classList.add('hidden');
    document.getElementById('victory').classList.remove('hidden');
    AudioSys.levelUp();
  },

  _spawnEnemy(){
    if(this.enemiesToSpawn<=0)return;
    const r=Math.random();
    // Boss triggers
    if(this.enemiesToSpawn===1&&!this.boss){
      let bTier=0;
      if(level===5) bTier=1;
      else if(level===7) bTier=2;
      else if(level===9) bTier=3;
      else if(level===10) bTier=4;
      if(bTier>0){ this.boss=new Boss(bTier); this.enemiesToSpawn=0; return; }
    }
    let type='basic';
    if(level>=2&&r<0.18) type='fast';
    if(level>=3&&r>0.87) type='heavy';
    if(level>=4&&r>0.62&&r<0.76) type='pulsar';
    if(level>=5&&r>0.42&&r<0.57) type='splitter';
    this.enemies.push(new Enemy(type));
    this.enemiesToSpawn--;
  },

  checkProgress(){
    if(this.boss&&!this.boss.active) this.boss=null;
    if(this.enemiesToSpawn===0&&!this.boss&&
       this.enemies.filter(e=>e.active).length===0){
      if(level>=10) this.victory();
      else this.nextLevel();
    }
  },

  nextLevel(){
    level++;
    this.bullets=[];
    this.enemiesToSpawn=14+level*9;
    this.waveTimer=0; this.spawnTimer=0; this.boss=null;
    this.spawnInterval=Math.max(42,105-level*6);
    this._schedulePowerups();
    updHUD(); AudioSys.levelUp();
    T.push(`LEVEL ${level}`,160,'#fa0');
    document.getElementById('boss-bar-wrap').style.display='none';
  },

  update(){
    frameCount++;
    this.starfield.update();
    if(gState!=='PLAYING')return;
    if(this.shake>0){ this.shake*=0.87; if(this.shake<0.3) this.shake=0; }
    if(this.darksMode>0) this.darksMode--;
    if(this.flatsFlash>0) this.flatsFlash--;
    if(this.setiMode>0){ this.setiMode--; this.setiX+=3; if(this.setiX>GW+70) this.setiX=-70; }
    if(comboTimer>0){ comboTimer--; if(comboTimer===0){
      combo=0; const cd=document.getElementById('combo-disp'); if(cd) cd.style.opacity=0;
    }}
    this.player.update(); T.update();
    this.waveTimer++; this.spawnTimer++;
    if(this.spawnTimer>this.spawnInterval&&this.waveTimer>130){ this._spawnEnemy(); this.spawnTimer=0; }
    for(let i=this.powerupQueue.length-1;i>=0;i--){
      if(this.waveTimer>=this.powerupQueue[i].time){
        this.powerups.push(new Powerup(this.powerupQueue[i].type));
        this.powerupQueue.splice(i,1);
      }
    }
    this.bullets=this.bullets.filter(b=>b.active);
    this.bullets.forEach(b=>b.update());
    this.powerups=this.powerups.filter(p=>p.active);
    this.powerups.forEach(p=>p.update());
    this.enemies=this.enemies.filter(e=>e.active);
    this.enemies.forEach(e=>e.update());
    this.particles=this.particles.filter(p=>p.life>0);
    this.particles.forEach(p=>p.update());
    if(this.boss&&this.boss.active) this.boss.update();
    this.novaRings=this.novaRings.filter(r=>r.active);
    this.novaRings.forEach(r=>r.update());

    // SETI mode — sweep destroys enemy bullets
    if(this.setiMode>0){
      this.bullets.filter(b=>!b.isPlayer).forEach(b=>{
        if(Math.abs(b.x-this.setiX)<55&&b.y>30){
          b.active=false;
          this.particles.push(new Particle(b.x,b.y,'#0ff',2.5,18));
        }
      });
    }
    // player bullets vs enemies + boss
    this.bullets.filter(b=>b.isPlayer).forEach(b=>{
      if(this.boss&&this.boss.active){
        // Try shield gens first
        if(this.boss.hitShieldGen(b)){ b.active=false; return; }
        if(rInt(b.getBounds(),this.boss.getBounds())){ b.active=false; this.boss.hit(1); return; }
      }
      this.enemies.forEach(e=>{
        if(b.active&&rInt(b.getBounds(),e.getBounds())&&e.y>0){
          b.active=false; e.hit(1);
          this.particles.push(new Particle(b.x,b.y,C.PB,2,12));
        }
      });
    });
    // powerup pickups
    this.powerups.forEach(p=>{
      if(rInt(p.getBounds(),this.player.getBounds())){
        p.active=false; AudioSys.powerup();
        if(p.type==='hp'){
          this.player.hp=Math.min(this.player.maxHp,this.player.hp+65); updHUD();
          T.push('HP RESTORED',90,'#f44');
        } else if(p.type==='fire'){
          this.player.fireRate=7;
          if(this.player.fireTimer) clearTimeout(this.player.fireTimer);
          this.player.fireTimer=setTimeout(()=>{ if(this.player) this.player.fireRate=13; },6500);
          T.push('DOUBLE FIRE — 6s',90,'#0af');
        } else if(p.type==='shield'){
          this.player.shield=Math.min(3,(this.player.shield||0)+1);
          T.push(`SHIELD +1 (${this.player.shield}/3)`,90,'#0ff');
          updHUD();
        } else if(p.type==='bomb'){
          this.enemies.forEach(e=>e.hit(999));
          if(this.boss) this.boss.hit(180);
          T.push('NOVA BOMB — ALL ENEMIES DESTROYED',130,'#fa0');
          this.triggerShake(22);
        }
      }
    });
    // enemy bullets vs player
    this.bullets.filter(b=>!b.isPlayer).forEach(b=>{
      if(rInt(b.getBounds(),this.player.getBounds())){ b.active=false; this.player.hit(20); }
    });
    // enemies vs player
    this.enemies.forEach(e=>{
      if(e.y>GH+45){ e.active=false; this.player.hit(15); this.checkProgress(); }
      else if(rInt(e.getBounds(),this.player.getBounds())){ e.hit(999); this.player.hit(35); }
    });
  },

  draw(){
    const bgs=['#000018','#001400','#141400','#140014','#140000',
               '#001414','#001200','#0f0e00','#0e000e','#0e0000'];
    ctx.fillStyle=bgs[(level-1)%bgs.length];
    ctx.fillRect(0,0,GW,GH);
    // flats flash
    if(this.flatsFlash>0){
      ctx.fillStyle=`rgba(255,255,255,${this.flatsFlash/35*0.72})`;
      ctx.fillRect(0,0,GW,GH);
    }
    ctx.save();
    if(this.shake>0) ctx.translate(rand(-this.shake,this.shake),rand(-this.shake,this.shake));
    this.starfield.draw(ctx);
    if(gState==='PLAYING'){
      this.powerups.forEach(p=>p.draw(ctx));
      // darks mode dims enemies
      if(this.darksMode>0){
        ctx.save(); ctx.globalAlpha=0.08;
        this.enemies.forEach(e=>e.draw(ctx));
        if(this.boss&&this.boss.active) this.boss.draw(ctx);
        ctx.restore();
      } else {
        this.enemies.forEach(e=>e.draw(ctx));
        if(this.boss&&this.boss.active) this.boss.draw(ctx);
      }
      this.bullets.forEach(b=>b.draw(ctx));
      this.particles.forEach(p=>p.draw(ctx));
      this.player.draw(ctx);
      this.novaRings.forEach(r=>r.draw(ctx));

      // SETI sweep beam
      if(this.setiMode>0){
        ctx.save();
        ctx.strokeStyle=`rgba(0,255,255,${0.6+0.3*Math.sin(frameCount*.2)})`;
        ctx.lineWidth=50; ctx.globalAlpha=0.12;
        ctx.beginPath(); ctx.moveTo(this.setiX,0); ctx.lineTo(this.setiX,GH); ctx.stroke();
        ctx.lineWidth=2; ctx.globalAlpha=0.7;
        ctx.beginPath(); ctx.moveTo(this.setiX,0); ctx.lineTo(this.setiX,GH); ctx.stroke();
        ctx.restore();
      }
    }
    T.draw(ctx);
    ctx.restore();
  }
};

// ─── HUD UPDATE ──────────────────────────────────────────────────────────────
function updHUD(){
  document.getElementById('score').textContent=score.toLocaleString();
  document.getElementById('level').textContent=level;
  if(G.player){
    const pct=Math.max(0,G.player.hp/G.player.maxHp*100);
    const fill=document.getElementById('hp-fill');
    fill.style.width=pct+'%';
    fill.style.background=pct>50?'#0f0':pct>25?'#fa0':'#f44';
    const si=document.getElementById('shield-ind');
    if(si) si.textContent=G.player.shield>0?'SHIELD: '+'█'.repeat(G.player.shield):'';
    const bi=document.getElementById('bomb-ind');
    if(bi) bi.textContent=G.bombStock>0?'BOMBS: '+'◈'.repeat(G.bombStock):'BOMBS: —';    
  }
}

// ─── GAME LOOP ───────────────────────────────────────────────────────────────
let lastMs=0;
function loop(ts){
  requestAnimationFrame(loop);
  if(!lastMs) lastMs=ts;
  if(ts-lastMs>INTERVAL){
    lastMs=ts-(ts-lastMs)%INTERVAL;
    G.update(); G.draw();
  }
}

G.init();
