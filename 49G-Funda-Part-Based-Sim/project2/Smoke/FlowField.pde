public class FlowField {
  PVector[][] vectors;
  int cols, rows;
  float grid_inc = 0.1;
  float noise_time_off = 0;
  float noise_time_inc = 0.002;
  int scl;
  
  FlowField(int res, float ntime_inc) {
    scl = res;
    noise_time_inc = ntime_inc;
    cols = floor(width / res) + 1;
    rows = floor(height / res) + 1;
    vectors = new PVector[cols] [rows];
  }
  
  void updateFF() {
    float xoff = 0;
    for (int y = 0; y < rows; y++) { 
      float yoff = 0;
      for (int x = 0; x < cols; x++) {
        float angle = noise(xoff, yoff, noise_time_off) * TWO_PI * 2;
        
                
        PVector v = PVector.fromAngle(angle);
        
        v.add(new PVector(0, -0.75));   
        
        
        if(x<cols/5){
          v.add(new PVector(0.6, 0));
        }else if (x > 4*cols/5){
          v.add(new PVector(-0.6, 0));
        }
       
        
        v.setMag(1);
        vectors[x][y] = v;
       
        xoff += grid_inc;
      }
      yoff += grid_inc;
    }
    noise_time_off += noise_time_inc;
  }
  
  
  void display() {
    for (int y = 0; y < rows; y++) { 
      for (int x = 0; x < cols; x++) {
        PVector v = vectors[x][y];
        
        stroke(0, 0, 0, 150);
        strokeWeight(1);
        pushMatrix();
        translate(x * scl, y * scl);
        rotate(v.heading());
        line(0, 0, scl, 0);
        popMatrix();
      }
    }
  }
}
