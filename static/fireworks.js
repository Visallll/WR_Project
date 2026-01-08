const canvas = document.getElementById("fireworks");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

class Firework {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = canvas.height;
        this.targetY = Math.random() * canvas.height / 2;
        this.speed = 4 + Math.random() * 3;
        this.exploded = false;
        this.particles = [];
        this.color = `hsl(${Math.random() * 360}, 100%, 60%)`;
    }

    update() {
        if (!this.exploded) {
            this.y -= this.speed;
            if (this.y <= this.targetY) {
                this.explode();
                this.exploded = true;
            }
        }

        this.particles.forEach(p => p.update());
    }

    draw() {
        if (!this.exploded) {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        this.particles.forEach(p => p.draw());
    }

    explode() {
        for (let i = 0; i < 40; i++) {
            this.particles.push(new Particle(this.x, this.y, this.color));
        }
    }
}

class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.angle = Math.random() * Math.PI * 2;
        this.speed = Math.random() * 5 + 2;
        this.life = 60;
        this.color = color;
    }

    update() {
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;
        this.life--;
    }

    draw() {
        ctx.fillStyle = this.color;
        ctx.globalAlpha = this.life / 60;
        ctx.beginPath();
        ctx.arc(this.x, this.y, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }
}

let fireworks = [];

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (Math.random() < 0.05) {
        fireworks.push(new Firework());
    }

    fireworks.forEach((fw, i) => {
        fw.update();
        fw.draw();
        if (fw.exploded && fw.particles.every(p => p.life <= 0)) {
            fireworks.splice(i, 1);
        }
    });

    requestAnimationFrame(animate);
}

animate();
