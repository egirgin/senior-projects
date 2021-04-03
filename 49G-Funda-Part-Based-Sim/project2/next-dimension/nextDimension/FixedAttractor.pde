class FixedAttractor{
  PVector position;
  float mass;


FixedAttractor(float xpos, float ypos, float amass){
  mass = amass;
  position = new PVector(xpos, ypos);
}


PVector getAttractionForceOn(Particle mp){
  PVector attrForce = PVector.sub(position, mp.pos);
  
  float distance = attrForce.mag();
  
  // F_g Magnitude
  float F_g_mag = (mass) / (distance*distance);
  
  attrForce.setMag(F_g_mag);
  
  return attrForce;
}


void display() {
    stroke(0);
    fill(175);
    ellipse(position.x, position.y, 2*sqrt(mass), 2*sqrt(mass));
  }
  

}
