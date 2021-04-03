FlowField flowfield;
ArrayList<Particle> particles;

boolean debug = false;

void setup() {
  size(1000, 1000, P2D);
  
  flowfield = new FlowField(10, 0.009);
  flowfield.updateFF();
  
  initNParticles(4000);
  
  background(0);
}

void draw() {
  flowfield.updateFF();
  
  if (debug) flowfield.display();
  
  for (Particle p : particles) {
    p.applyFieldForce(flowfield);
    p.run();
  }
}

// This is how we init particles 
void initNParticles(int n){
  particles = new ArrayList<Particle>();
  for (int i = 0; i < n; i++) {
    float maxSpeed = random(1, 7);
    
    if (i <= n*(0.20)){
      PVector start_point = new PVector(random(width/5)+width/2-width/5, height);
      particles.add(new Particle(start_point, maxSpeed, 155, 156, 152));
    }
    else if (i <= n*(0.21)){
      PVector start_point = new PVector(random(width/5)+width/2-width/5, height);
      particles.add(new Particle(start_point, maxSpeed, 249, 250, 222));
    }
    else{
      PVector start_point = new PVector(random(width/5)+width/2-width/5, height);
      particles.add(new Particle(start_point, maxSpeed, 53, 54, 52));
    }
    
  }
}
