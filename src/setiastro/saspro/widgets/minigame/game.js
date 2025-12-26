/**
 * Neon Invaders: Hyperdrive
 * Core Game Engine
 */

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// --- CONSTANTS ---
const GAME_WIDTH = 800;
const GAME_HEIGHT = 600;
const FPS = 48; // Reduced to 80% speed (Original 60)
const DT = 1 / FPS;

const COLORS = {
    PLAYER: '#0ff',
    PLAYER_BULLET: '#0ff',
    ENEMY_BASIC: '#f0f',
    ENEMY_FAST: '#0f0',
    ENEMY_HEAVY: '#f00',
    ENEMY_BOSS: '#fff',
    ENEMY_BULLET: '#f00',
    PARTICLE_EXPLOSION: '#fa0',
    PARTICLE_THRUST: '#0ff',
    POWERUP_HP: '#f00',
    POWERUP_HP: '#f00',
    POWERUP_FIRE: '#0ff',
    POWERUP_SHIELD: '#00ffff' // Cyan
};

const KEYS = {
    LEFT: 'ArrowLeft',
    RIGHT: 'ArrowRight',
    UP: 'ArrowUp',
    DOWN: 'ArrowDown',
    SHOOT: ' '
};

// --- GLOBAL STATE ---
let gameState = 'MENU'; // MENU, PLAYING, GAMEOVER, VICTORY, LEVEL_SELECT
let lastTime = 0;
let score = 0;
let level = 1;

// --- INPUT HANDLER ---
const Input = {
    keys: {},
    init() {
        window.addEventListener('keydown', e => this.keys[e.key] = true);
        window.addEventListener('keyup', e => this.keys[e.key] = false);
    },
    isDown(key) { return this.keys[key]; }
};
Input.init();

// --- UTILS ---
const dist = (x1, y1, x2, y2) => Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
const rectIntersect = (r1, r2) => !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom || r2.bottom < r1.top);
const rand = (min, max) => Math.random() * (max - min) + min;

// --- AUDIO (Placeholder for synth) ---
const AudioSys = {
    ctx: new (window.AudioContext || window.webkitAudioContext)(),
    playTone(freq, type, duration) {
        if (this.ctx.state === 'suspended') this.ctx.resume();
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + duration);
    },
    shoot() { this.playTone(400, 'square', 0.1); },
    explosion() { this.playTone(100, 'sawtooth', 0.3); },
    hit() { this.playTone(200, 'sawtooth', 0.1); }
};

// --- CLASSES ---

class Starfield {
    constructor() {
        this.stars = [];
        this.speed = 2; // Base speed
        for (let i = 0; i < 100; i++) this.addStar(true);
    }

    addStar(randomY = false) {
        this.stars.push({
            x: rand(0, GAME_WIDTH),
            y: randomY ? rand(0, GAME_HEIGHT) : -10,
            z: rand(0.5, 2), // Depth factor for parallax
            size: rand(0.5, 2)
        });
    }

    update() {
        // Speed locked (User Request: "same as level 1")
        const currentSpeed = this.speed;
        this.stars.forEach(s => {
            s.y += currentSpeed * s.z;
            if (s.y > GAME_HEIGHT) {
                s.y = -10;
                s.x = rand(0, GAME_WIDTH);
            }
        });
    }

    draw(ctx) {
        // Change Star color based on level - More saturated/distinct
        const colors = ['#ffffff', '#00ff00', '#ffff00', '#ff00ff', '#ff0000', '#00ffff', '#00ff00', '#ffff00', '#ff00ff', '#ff0000'];
        ctx.fillStyle = colors[(level - 1) % colors.length] || '#fff';

        this.stars.forEach(s => {
            ctx.globalAlpha = s.z / 2;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;
    }
}

class Particle {
    constructor(x, y, color, speed, life) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.life = life;
        this.maxLife = life;
        const angle = rand(0, Math.PI * 2);
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        this.size = rand(1, 3);
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life--;
        this.size *= 0.95;
    }
    draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.globalAlpha = this.life / this.maxLife;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }
}

class Powerup {
    constructor(type) {
        this.type = type; // 'hp' or 'fire'
        this.x = rand(50, GAME_WIDTH - 50);
        this.y = -30;
        this.width = 30;
        this.height = 30;
        this.speed = 2;
        this.active = true;
        this.speed = 2;
        this.active = true;
        this.color = type === 'hp' ? COLORS.POWERUP_HP : (type === 'fire' ? COLORS.POWERUP_FIRE : COLORS.POWERUP_SHIELD);
    }

    update() {
        this.y += this.speed;
        if (this.y > GAME_HEIGHT + 30) this.active = false;
    }

    draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        if (this.type === 'hp') {
            // Heart shape approximation or Cross
            ctx.fillRect(this.x - 10, this.y - 4, 20, 8);
            ctx.fillRect(this.x - 4, this.y - 10, 8, 20);
        } else {
            // Lightning bolt / Energy
            ctx.moveTo(this.x, this.y - 15);
            ctx.lineTo(this.x + 10, this.y);
            ctx.lineTo(this.x - 5, this.y);
            ctx.lineTo(this.x + 5, this.y + 15);
            ctx.lineTo(this.x - 10, this.y);
            ctx.lineTo(this.x + 5, this.y);
            ctx.closePath();
            ctx.fill();
        }
        ctx.shadowBlur = 0;
        ctx.fill();
    }

    getBounds() {
        return { left: this.x - 15, right: this.x + 15, top: this.y - 15, bottom: this.y + 15 };
    }
}

class Bullet {
    constructor(x, y, vy, isPlayer) {
        this.x = x;
        this.y = y;
        this.vy = vy;
        this.isPlayer = isPlayer;
        this.width = 4;
        this.height = 10;
        this.active = true;
    }

    update() {
        this.y += this.vy;
        if (this.y < -50 || this.y > GAME_HEIGHT + 50) this.active = false;
    }

    draw(ctx) {
        ctx.fillStyle = this.isPlayer ? COLORS.PLAYER_BULLET : COLORS.ENEMY_BULLET;
        ctx.shadowBlur = 5;
        ctx.shadowColor = ctx.fillStyle;
        ctx.fillRect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height);
        ctx.shadowBlur = 0;
    }

    getBounds() {
        return { left: this.x - 2, right: this.x + 2, top: this.y - 5, bottom: this.y + 5 };
    }
}

class Player {
    constructor() {
        this.width = 40;
        this.height = 40;
        this.x = GAME_WIDTH / 2;
        this.y = GAME_HEIGHT - 100;
        this.speed = 5;
        this.hp = 100;
        this.maxHp = 100;
        this.cooldown = 0;
        this.fireRate = 10;
        this.fireRate = 10;
        this.fireRateTimer = null; // Track timeout ID
        this.dead = false;
        this.shield = 0; // Shield strength (0 or 1)
    }

    update() {
        if (this.dead) return;

        // Movement
        let dx = 0;
        let dy = 0;
        if (Input.isDown(KEYS.LEFT)) dx -= this.speed;
        if (Input.isDown(KEYS.RIGHT)) dx += this.speed;
        if (Input.isDown(KEYS.UP)) dy -= this.speed;
        if (Input.isDown(KEYS.DOWN)) dy += this.speed;

        this.x += dx;
        this.y += dy;

        // Clamp
        this.x = Math.max(this.width / 2, Math.min(GAME_WIDTH - this.width / 2, this.x));
        this.y = Math.max(this.height / 2, Math.min(GAME_HEIGHT - this.height / 2, this.y));

        // Shoot
        if (this.cooldown > 0) this.cooldown--;
        if (Input.isDown(KEYS.SHOOT) && this.cooldown <= 0) {
            Game.bullets.push(new Bullet(this.x, this.y - 20, -10, true));
            if (this.fireRate < 10) { // Double shot for fast fire
                setTimeout(() => Game.bullets.push(new Bullet(this.x, this.y - 20, -10, true)), 100);
            }
            AudioSys.shoot();
            this.cooldown = this.fireRate;
        }

        // Thruster particles
        if (Math.random() < 0.5) {
            Game.particles.push(new Particle(this.x + rand(-5, 5), this.y + 20, COLORS.PARTICLE_THRUST, 1, 20));
        }
    }

    draw(ctx) {
        if (this.dead) return;
        ctx.strokeStyle = COLORS.PLAYER;
        ctx.lineWidth = 2;
        ctx.shadowColor = COLORS.PLAYER;
        ctx.shadowBlur = 10;

        // Draw Ship (Triangle shape)
        ctx.beginPath();
        ctx.moveTo(this.x, this.y - 20); // Nose
        ctx.lineTo(this.x + 20, this.y + 20); // Right Wing
        ctx.lineTo(this.x, this.y + 10); // Center Engine
        ctx.lineTo(this.x - 20, this.y + 20); // Left Wing
        ctx.closePath();
        ctx.stroke();

        ctx.fillStyle = COLORS.PLAYER;
        ctx.fill();

        ctx.shadowBlur = 0;
    }

    hit(damage) {
        if (this.shield > 0) {
            this.shield--; // Absorb hit
            AudioSys.hit(); // Maybe different sound?
            return;
        }
        this.hp -= damage;
        AudioSys.hit();
        updateHUD();
        if (this.hp <= 0) {
            this.hp = 0;
            this.dead = true;
            Game.createExplosion(this.x, this.y, 50, COLORS.PLAYER);
            setTimeout(() => Game.gameOver(), 1000);
        }
    }

    getBounds() {
        return { left: this.x - 15, right: this.x + 15, top: this.y - 15, bottom: this.y + 15 };
    }
}

class Enemy {
    constructor(type) {
        this.type = type;
        this.width = 30;
        this.height = 30;
        this.x = rand(50, GAME_WIDTH - 50);
        this.y = -50;
        this.active = true;
        this.timer = Math.floor(rand(0, 100)); // Randomize start timer so they don't sync
        this.lastShot = 0; // frame count


        // Define stats based on type
        switch (type) {
            case 'basic':
                this.hp = 2; // User requested 2 hits
                this.score = 100;
                this.speed = 1.5;
                this.color = COLORS.ENEMY_BASIC;
                this.shootInterval = 3 * 60; // 3 seconds (was 5)
                break;
            case 'fast':
                this.hp = 1;
                this.score = 50; // User requested 50
                this.speed = 3;
                this.width = 20; this.height = 20;
                this.color = COLORS.ENEMY_FAST;
                this.shootInterval = 0; // Doesn't shoot usually
                break;
            case 'heavy':
                this.hp = 5; // Reduced from 10 to 5 (User Request)
                this.speed = 0.8;
                this.score = 250; // User requested 250
                this.width = 50; this.height = 50;
                this.color = COLORS.ENEMY_HEAVY;
                this.shootInterval = 3 * 60; // 3 seconds
                break;
        }
    }

    update() {
        this.y += this.speed;
        this.timer++;

        // Shooting logic replacement
        // "I nemici non sparano se non arrivano ad un certo punto" -> Check if y > 50 (fully on screen)
        if (this.shootInterval > 0 && this.timer % this.shootInterval === 0 && this.y > 50 && this.y < GAME_HEIGHT - 50) {
            Game.bullets.push(new Bullet(this.x, this.y + this.height / 2, 5, false));
        }

        // Basic AI movement
        if (this.type === 'fast') {
            this.x += Math.sin(this.timer * 0.1) * 2;
        }

        // Random erratic shots for basic rarely? No, strict interval now. 
        // Removing old random shooting logic.

        // Constrain X
        if (this.x < 0) this.x = 0;
        if (this.x > GAME_WIDTH) this.x = GAME_WIDTH;
    }

    draw(ctx) {
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2;
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 10;

        ctx.beginPath();
        if (this.type === 'basic') {
            ctx.rect(this.x - 15, this.y - 15, 30, 30);
        } else if (this.type === 'fast') {
            ctx.moveTo(this.x, this.y + 10);
            ctx.lineTo(this.x + 10, this.y - 10);
            ctx.lineTo(this.x - 10, this.y - 10);
            ctx.closePath();
        } else if (this.type === 'heavy') {
            ctx.arc(this.x, this.y, 25, 0, Math.PI * 2);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    hit(damage) {
        if (this.y < 0) return; // Cannot hit off-screen enemies
        this.hp -= damage;
        AudioSys.hit();
        if (this.hp <= 0) {
            this.active = false;
            score += this.score;
            Game.createExplosion(this.x, this.y, 20, this.color);
            updateHUD();
            Game.checkLevelProgress();
        }
    }

    getBounds() {
        let w = this.width / 2;
        let h = this.height / 2;
        return { left: this.x - w, right: this.x + w, top: this.y - h, bottom: this.y + h };
    }
}

class Boss {
    constructor() {
        this.width = 120;
        this.height = 80;
        this.x = GAME_WIDTH / 2;
        this.y = -100;
        this.active = true;
        this.hp = 500;
        this.maxHp = 500;
        this.speed = 2;
        this.timer = 0;
        this.dir = 1;
        this.color = COLORS.ENEMY_BOSS;
        this.score = 5000;
    }

    update() {
        this.timer++;

        // Entrance
        if (this.y < 100) {
            this.y += 1;
        } else {
            // Horizontal Movement
            this.x += this.speed * this.dir;
            if (this.x > GAME_WIDTH - 80 || this.x < 80) this.dir *= -1;
        }

        // Attacks
        // 1. Triple Shot (Front)
        if (this.timer % 60 === 0) {
            Game.bullets.push(new Bullet(this.x, this.y + 40, 5, false));
            Game.bullets.push(new Bullet(this.x - 30, this.y + 30, 5, false));
            Game.bullets.push(new Bullet(this.x + 30, this.y + 30, 5, false));
        }

        // 2. Spread Shot (every 3s)
        if (this.timer % 180 === 0) {
            for (let i = -2; i <= 2; i++) {
                // Fake angle by setting vx
                let b = new Bullet(this.x, this.y + 40, 4, false);
                b.x += i * 15; // Offset start
                b.width = 8;
                Game.bullets.push(b);
            }
        }
    }

    draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 20;

        // Custom shape for Mothership
        ctx.beginPath();
        ctx.moveTo(this.x, this.y + 40);
        ctx.lineTo(this.x + 60, this.y - 20);
        ctx.lineTo(this.x + 30, this.y - 40);
        ctx.lineTo(this.x - 30, this.y - 40);
        ctx.lineTo(this.x - 60, this.y - 20);
        ctx.closePath();
        ctx.fill();

        // Reactor Core
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(this.x, this.y, 15, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowBlur = 0;

        // HP Bar (Above Boss)
        ctx.fillStyle = '#555';
        ctx.fillRect(this.x - 50, this.y - 60, 100, 10);
        ctx.fillStyle = '#f00';
        ctx.fillRect(this.x - 50, this.y - 60, 100 * (this.hp / this.maxHp), 10);
    }

    hit(damage) {
        this.hp -= damage;
        AudioSys.hit();
        if (this.hp <= 0) {
            this.active = false;
            score += this.score;
            Game.createExplosion(this.x, this.y, 100, '#fff');
            updateHUD();
            Game.victory(); // Win game after boss
        }
    }

    getBounds() {
        return { left: this.x - 50, right: this.x + 50, top: this.y - 30, bottom: this.y + 30 };
    }
}

// --- GAME MANAGER ---
const Game = {
    player: null,
    bullets: [],
    enemies: [],
    particles: [],
    powerups: [],
    starfield: null,
    waveTimer: 0,
    enemiesToSpawn: 0,
    spawnTimer: 0,
    powerupQueue: [],
    starfield: null,
    waveTimer: 0,
    enemiesToSpawn: 0,
    spawnTimer: 0,

    init() {
        canvas.width = GAME_WIDTH;
        canvas.height = GAME_HEIGHT;
        this.starfield = new Starfield();
        this.setupUI();
        requestAnimationFrame(loop);
    },

    shake: 0,
    triggerShake(amount) {
        this.shake = amount;
    },

    startLevel(startLevel) {
        level = startLevel;
        if (!this.player) this.player = new Player();
        this.bullets = [];
        this.enemies = [];
        this.particles = [];
        this.powerups = [];
        this.enemiesToSpawn = 10 * level;
        this.waveTimer = 0;

        // Initialize Counts (Base: 1 Fire, 1 HP)
        let hpCount = 1;
        let utilCount = 1;

        if (level >= 3) hpCount++;
        if (level >= 6) { hpCount += 2; utilCount += 2; } // L6-8: 4 HP, 3 Util
        if (level >= 9) { hpCount += 1; utilCount += 1; } // L9-10: 5 HP, 4 Util

        // 1. Utility
        this.powerupQueue = [];
        let timeOffset = 180 + rand(100, 300);
        this.powerupQueue.push({ type: 'fire', time: timeOffset });

        if (level >= 2) {
            for (let i = 0; i < utilCount - 1; i++) {
                timeOffset += rand(150, 400); // Faster intervals
                this.powerupQueue.push({ type: 'shield', time: timeOffset });
            }
        }

        // 2. HP
        for (let i = 0; i < hpCount; i++) {
            timeOffset += rand(150, 400); // Faster intervals
            this.powerupQueue.push({ type: 'hp', time: timeOffset });
        }
    },

    start(startLevel = 1) {
        level = startLevel;
        score = 0;
        this.player = new Player();
        this.bullets = [];
        this.enemies = [];
        this.particles = [];
        this.powerups = [];
        this.enemiesToSpawn = 10 * level; // Simple scaling

        // Schedule powerups (frame count)
        this.powerupQueue = [];
        let timeOffset = rand(300, 600); // Start spawning after 5-10s

        // Initialize Counts (Base: 1 Fire, 1 HP)
        let hpCount = 1;
        let utilCount = 1;

        if (level >= 3) hpCount++;
        if (level >= 6) { hpCount += 2; utilCount += 2; } // L6-8: 4 HP, 3 Util
        if (level >= 9) { hpCount += 1; utilCount += 1; } // L9-10: 5 HP, 4 Util

        // 1. Utility Powerups (Fire / Shield)
        // Fire always 1
        this.powerupQueue.push({ type: 'fire', time: timeOffset });

        // Remaining Utility slots used for Shield (if Level >= 2)
        if (level >= 2) {
            // We used 1 for Fire, so loop remaining utilCount - 1
            for (let i = 0; i < utilCount - 1; i++) {
                timeOffset += rand(150, 400); // Faster intervals
                this.powerupQueue.push({ type: 'shield', time: timeOffset });
            }
        }

        // 2. HP Powerups
        for (let i = 0; i < hpCount; i++) {
            timeOffset += rand(150, 400); // Faster intervals
            this.powerupQueue.push({ type: 'hp', time: timeOffset });
        }

        gameState = 'PLAYING';

        document.getElementById('main-menu').classList.add('hidden');
        document.getElementById('level-select').classList.add('hidden');
        document.getElementById('game-over').classList.add('hidden');
        document.getElementById('victory').classList.add('hidden');
        document.getElementById('hud').classList.remove('hidden');
        updateHUD();
    },

    setupUI() {
        // Main Menu
        document.getElementById('btn-start').onclick = () => this.start(1);
        document.getElementById('btn-levels').onclick = () => {
            document.getElementById('main-menu').classList.add('hidden');
            document.getElementById('level-select').classList.remove('hidden');
            this.renderLevelGrid();
        };
        // Level Select
        document.getElementById('btn-back').onclick = () => {
            document.getElementById('level-select').classList.add('hidden');
            document.getElementById('main-menu').classList.remove('hidden');
        };
        // Game Over
        document.getElementById('btn-retry').onclick = () => this.start(level);
        document.getElementById('btn-menu').onclick = () => this.showMenu();
        // Victory
        document.getElementById('btn-menu-win').onclick = () => this.showMenu();
    },

    renderLevelGrid() {
        const grid = document.getElementById('level-grid');
        grid.innerHTML = '';
        for (let i = 1; i <= 10; i++) {
            const btn = document.createElement('button');
            btn.className = 'level-btn';
            btn.innerText = `LEVEL ${i}`;
            btn.onclick = () => this.start(i);
            grid.appendChild(btn);
        }
    },

    showMenu() {
        gameState = 'MENU';
        document.getElementById('game-over').classList.add('hidden');
        document.getElementById('victory').classList.add('hidden');
        document.getElementById('hud').classList.add('hidden');
        document.getElementById('main-menu').classList.remove('hidden');
    },

    gameOver() {
        gameState = 'GAMEOVER';
        document.getElementById('final-score').innerText = score;
        document.getElementById('game-over').classList.remove('hidden');
    },

    victory() {
        gameState = 'VICTORY';
        document.getElementById('victory-score').innerText = score;
        document.getElementById('victory').classList.remove('hidden');
    },

    spawnPowerup(type) {
        this.powerups.push(new Powerup(type));
    },

    showLevelTransition() {
        // Warp Effect Removed
        // Text overlay
        const t = document.createElement('div');
        t.innerText = `LEVEL ${level}`;
        t.innerText += `\nGET READY`;
        t.style.textAlign = 'center';
        t.className = 'level-transition';
        t.style.position = 'absolute';
        t.style.top = '50%';
        t.style.left = '50%';
        t.style.transform = 'translate(-50%, -50%)';
        t.style.fontSize = '4rem';
        t.style.color = '#fff';
        t.style.textShadow = `0 0 20px ${this.getLevelColor()}`;
        t.style.fontWeight = 'bold';
        t.style.zIndex = '100';
        t.style.animation = 'fadeUp 2s forwards';
        document.body.appendChild(t);

        const originalSpeed = this.starfield.speed;
        // User requested removing warp speed effect
        // this.starfield.speed = 20; 
        setTimeout(() => {
            t.remove();
            // this.starfield.speed = originalSpeed;
        }, 2000);
    },

    getLevelColor() {
        // Enhanced colors for text shadow
        const colors = ['#00ffff', '#00ff00', '#ffff00', '#ff00ff', '#ff0000', '#00ffff', '#00ff00', '#ffff00', '#ff00ff', '#ff0000'];
        return colors[(level - 1) % colors.length];
    },

    createExplosion(x, y, count, color) {
        AudioSys.explosion();
        this.triggerShake(count / 2);
        for (let i = 0; i < count; i++) {
            this.particles.push(new Particle(x, y, color, rand(1, 4), rand(20, 40)));
        }
    },

    spawnEnemy() {
        if (this.enemiesToSpawn <= 0) return;

        // Spawn logic based on level
        let type = 'basic';
        const r = Math.random();

        if (level === 2) {
            if (r < 0.2) type = 'fast';
        } else if (level >= 3) {
            if (r < 0.3) type = 'fast';
            if (r > 0.9) type = 'heavy';
        } else if (level === 5 && this.enemiesToSpawn === 1) {
            // Boss Spawn (Mid-point)
            this.enemies.push(new Boss());
            this.enemiesToSpawn--;
            return;
        } else if (level >= 6 && level < 10) {
            // Harder scaling for 6-9
            if (r < 0.4) type = 'fast';
            if (r > 0.8) type = 'heavy';
        } else if (level === 10 && this.enemiesToSpawn === 1) {
            // Final Boss
            let boss = new Boss();
            boss.hp = 1000; // Double HP
            boss.maxHp = 1000;
            boss.color = '#fff'; // White Boss
            this.enemies.push(boss);
            this.enemiesToSpawn--;
            return;
        }

        this.enemies.push(new Enemy(type));
        this.enemiesToSpawn--;
    },

    checkLevelProgress() {
        if (this.enemiesToSpawn === 0 && this.enemies.filter(e => e.active).length === 0) {
            // Next Level
            if (level === 10) {
                this.victory();
            } else {
                this.nextLevel();
            }
        }
    },

    nextLevel() {
        level++;
        this.enemiesToSpawn = 10 * level + 5;
        this.waveTimer = 0;

        // Initialize Counts (Base: 1 Fire, 1 HP)
        let hpCount = 1;
        let utilCount = 1;

        if (level >= 3) hpCount++;
        if (level >= 6) { hpCount += 2; utilCount += 2; } // L6-8: 4 HP, 3 Util
        if (level >= 9) { hpCount += 1; utilCount += 1; } // L9-10: 5 HP, 4 Util

        // 1. Utility
        this.powerupQueue = [];
        let timeOffset = 180 + rand(100, 300);
        this.powerupQueue.push({ type: 'fire', time: timeOffset });

        if (level >= 2) {
            for (let i = 0; i < utilCount - 1; i++) {
                timeOffset += rand(150, 400); // Faster intervals
                this.powerupQueue.push({ type: 'shield', time: timeOffset });
            }
        }

        // 2. HP
        for (let i = 0; i < hpCount; i++) {
            timeOffset += rand(150, 400); // Faster intervals
            this.powerupQueue.push({ type: 'hp', time: timeOffset });
        }



        this.showLevelTransition();
        updateHUD();
    },

    update() {
        if (gameState !== 'PLAYING') {
            this.starfield.update(); // Keep stars moving in menu
            return;
        }

        this.starfield.update();
        this.player.update();

        // Spawning
        this.spawnTimer++;

        // Delay spawning at start of level (3 seconds = 180 frames) or if enemiesToSpawn is empty
        if (this.waveTimer >= 180) {
            // Faster spawn rate: Cap at Level 5
            // Level 1: 90 - 8 = 82 frames.
            // Level 5: 90 - 40 = 50 frames.
            // Level 10: Still 50 frames (using Math.min)
            const effectiveLevel = Math.min(level, 5);
            if (this.spawnTimer > 90 - (effectiveLevel * 8)) {
                this.spawnEnemy();
                this.spawnTimer = 0;
            }
        }

        // Powerup Spawning
        this.waveTimer++;
        for (let i = this.powerupQueue.length - 1; i >= 0; i--) {
            if (this.waveTimer >= this.powerupQueue[i].time) {
                this.spawnPowerup(this.powerupQueue[i].type);
                this.powerupQueue.splice(i, 1);
            }
        }

        // Entities
        this.bullets = this.bullets.filter(b => b.active);
        this.bullets.forEach(b => b.update());

        this.powerups = this.powerups.filter(p => p.active);
        this.powerups.forEach(p => p.update());

        this.enemies = this.enemies.filter(e => e.active);
        this.enemies.forEach(e => e.update());

        this.particles = this.particles.filter(p => p.life > 0);
        this.particles.forEach(p => p.update());

        // Collisions
        // Player Bullets -> Enemies
        this.bullets.filter(b => b.isPlayer).forEach(b => {
            this.enemies.forEach(e => {
                if (rectIntersect(b.getBounds(), e.getBounds()) && e.y > 0) { // Check visibility
                    b.active = false;
                    e.hit(Game.player.fireRate < 10 ? 2 : 1); // Double damage if fast fire? Or just more bullets. Keep damage 1 but fire faster.
                    Game.createExplosion(b.x, b.y, 5, b.isPlayer ? COLORS.PLAYER_BULLET : COLORS.ENEMY_BULLET);
                }
            });
        });

        // Player -> Powerups
        this.powerups.forEach(p => {
            if (rectIntersect(p.getBounds(), this.player.getBounds())) {
                p.active = false;
                if (p.type === 'hp') {
                    this.player.hp = this.player.maxHp; // Full Restore
                    updateHUD();
                } else if (p.type === 'fire') {
                    this.player.fireRate = 5;
                    // Clear existing timer if any to extend duration instead of cutting it short
                    if (this.player.fireRateTimer) clearTimeout(this.player.fireRateTimer);
                    this.player.fireRateTimer = setTimeout(() => {
                        if (this.player) {
                            this.player.fireRate = 10;
                            this.player.fireRateTimer = null;
                        }
                    }, 5000);
                } else if (p.type === 'shield') {
                    this.player.shield = 1;
                }
            }
        });

        // Enemy Bullets -> Player
        this.bullets.filter(b => !b.isPlayer).forEach(b => {
            if (rectIntersect(b.getBounds(), this.player.getBounds())) {
                b.active = false;
                this.player.hit(20);
            }
        });

        // Enemies -> Player (Crash) or Escaped
        this.enemies.forEach(e => {
            if (e.y > GAME_HEIGHT + 30) {
                e.active = false;
                // Penalty for letting enemy pass
                this.player.hit(10);
                Game.checkLevelProgress();
            } else if (rectIntersect(e.getBounds(), this.player.getBounds())) {
                e.hit(100); // Enemy dies
                this.player.hit(30); // Player takes damage
            }
        });
    },

    draw() {
        // Clear - Background Color changes significantly per level
        // Clear - Background Color changes significantly per level
        // More visible changes: Dark Blue, Dark Green, Dark Olive, Dark Purple, Dark Red
        const bgColors = ['#000022', '#002200', '#222200', '#220022', '#220000', '#000022', '#002200', '#222200', '#220022', '#220000'];
        ctx.fillStyle = bgColors[(level - 1) % bgColors.length];
        ctx.fillRect(0, 0, GAME_WIDTH, GAME_HEIGHT);

        ctx.save();
        if (this.shake > 0) {
            const dx = Math.random() * this.shake - this.shake / 2;
            const dy = Math.random() * this.shake - this.shake / 2;
            ctx.translate(dx, dy);
            this.shake *= 0.9;
            if (this.shake < 0.5) this.shake = 0;
        }

        // Stars
        this.starfield.draw(ctx);

        if (gameState === 'PLAYING') {
            this.player.draw(ctx);
            this.powerups.forEach(p => p.draw(ctx));
            this.enemies.forEach(e => e.draw(ctx));
            this.bullets.forEach(b => b.draw(ctx));
            this.particles.forEach(p => p.draw(ctx));
        }
        ctx.restore();
    }
};

function updateHUD() {
    document.getElementById('score').innerText = score;
    document.getElementById('level').innerText = level;
    if (Game.player) {
        const pct = (Game.player.hp / Game.player.maxHp) * 100;
        document.getElementById('hp-fill').style.width = `${Math.max(0, pct)}%`;
    }
}

// Game Loop
let lastTimeMs = 0;
const FRAME_INTERVAL = 1000 / FPS;

function loop(timestamp) {
    requestAnimationFrame(loop);

    if (!lastTimeMs) lastTimeMs = timestamp;
    const elapsed = timestamp - lastTimeMs;

    if (elapsed > FRAME_INTERVAL) {
        lastTimeMs = timestamp - (elapsed % FRAME_INTERVAL);
        Game.update();
        Game.draw();
    }
}

// Start
Game.init();
window.Game = Game; // Expose for debugging
