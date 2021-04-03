public class Particle {
  PVector pos;
  PVector vel;
  PVector acc;
  PVector previousPos;
  float maxSpeed;
  int r, g, b;
   
  Particle(PVector start, float maxspeed, int pr, int pg, int pb) {
    maxSpeed = maxspeed;
    pos = start;
    vel = new PVector(0, 0);
    acc = new PVector(0, 0);
    previousPos = pos.copy();
    r = pr;
    g = pg;
    b = pb;
  }
  
  void run() {
    updatePosition();
    edges();
    show();
  }
  
  void updatePosition() {
    pos.add(vel);
    vel.add(acc);
    vel.limit(maxSpeed);
    acc.mult(0);
  }
  
  void applyForce(PVector force) {
    acc.add(force); 
  }
  
  void applyFieldForce(FlowField flowfield) {
    int x = floor(pos.x / flowfield.scl);
    int y = floor(pos.y / flowfield.scl);
    
    PVector force = flowfield.vectors[x][y];
    applyForce(force);
  }
  
  void show() {
    if (r == 249){
      stroke(r, g, b, 255);
      strokeWeight(1);
    }
    else{
    stroke(r, g, b, 25);
    strokeWeight(2);
    }
    //strokeWeight(2);
    line(pos.x, pos.y, previousPos.x, previousPos.y);
    point(pos.x, pos.y);
    updatePreviousPos();
  }
  
  void updatePreviousPos() {
    this.previousPos.x = pos.x;
    this.previousPos.y = pos.y;
  }
  
  void edges() {
    if (pos.x > width) {
      pos.x = 0;
      updatePreviousPos();
    }
    if (pos.x < 0) {
      pos.x = width;    
      updatePreviousPos();
    }
    if (pos.y > height) {
      pos.y = height;
      updatePreviousPos();
    }
    if (pos.y < 0) {
      pos.y = 0;
      updatePreviousPos();
    }
  }
  
}
