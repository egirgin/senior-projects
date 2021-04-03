ArrayList<Particle> particles;
FixedAttractor fa1;

boolean debug = false;
int step = 0;

float f1offx = 0;

float f1offy = -5;

void setup() {
  size(1000, 1000, P2D);
  fa1 = new FixedAttractor(width/2, height/2, 5000);

  
  initNParticles(5000);
  
  background(0);
}

void draw() {
  step ++;
  
  
  
  if (step%10 == 0){
    
    f1offx += 0.01;
    f1offy -= 0.01;
    fa1.position = new PVector(noise(f1offx)*width, noise(f1offy)*height);
    
  }
  
  
  
  for (Particle p : particles) {
    PVector attraction_force1 = fa1.getAttractionForceOn(p);
    p.applyForce(attraction_force1);
    p.run();
  }
}

// This is how we init particles 
void initNParticles(int n){
  particles = new ArrayList<Particle>();
  for (int i = 0; i < n; i++) {
    float maxSpeed = random(1, 1);
    
    
    if (i%4 == 0){
      PVector start_point = new PVector(random(width), 0);
      particles.add(new Particle(start_point, maxSpeed, 100, 50, 200));
    }
    else if (i%4 == 1){
      PVector start_point = new PVector(0, random(height));
      particles.add(new Particle(start_point, maxSpeed, 200, 0, 100));
    }
    else if (i%4 == 2){
      PVector start_point = new PVector(width, random(height));
      particles.add(new Particle(start_point, maxSpeed, 200, 0, 100));
    }
    else{
      PVector start_point = new PVector(random(width), height);
      particles.add(new Particle(start_point, maxSpeed, 100, 50, 200));
    }
    
  }
}
