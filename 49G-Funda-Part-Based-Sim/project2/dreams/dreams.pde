FlowField flowfield;
ArrayList<Particle> particles;

boolean debug = false;


int step = 0;

void setup() {
  size(1000, 1000, P2D);
  
  flowfield = new FlowField(10, 0.009);
  flowfield.updateFF();
  
  initNParticles(2500);
  
  background(0);
}

void draw() {
  step ++;

  if (step%50 == 0){
    flowfield.updateFF();
  }

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
    float maxSpeed = random(1, 3);
    
    if (i < n*(0.33)){
      PVector start_point = new PVector(0, 0);
      particles.add(new Particle(start_point, maxSpeed, 247, 15, 81));
    }
    else if(i < n*(0.66)){
      PVector start_point = new PVector(0, 0);
      particles.add(new Particle(start_point, maxSpeed, 209, 180, 188));
    }
    else{
      PVector start_point = new PVector(0, 0);
      particles.add(new Particle(start_point, maxSpeed, 237, 226, 126));
    }
    
  }
}
