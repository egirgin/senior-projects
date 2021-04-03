FlowField flowfield;
ArrayList<Particle> particles;

boolean debug = false;

void setup() {
  size(1000, 1000, P2D);
  
  flowfield = new FlowField(20, 0.01);
  flowfield.updateFF();
  
  initNParticles(10000);
  
  background(0);
}

void draw() {
  //background(0); // Comment out to see the effect
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
    float maxSpeed = random(1, 2);
    
    if(i%2 == 0){
      PVector start_point = new PVector(random(width), height/2);
      particles.add(new Particle(start_point, maxSpeed, 227, 100, 100));
    }
    else{
      PVector start_point = new PVector(random(width), height/2);
      particles.add(new Particle(start_point, maxSpeed, 66, 255, 211));
    }
    
  }
}
