class Person{

  int column, row;
  int cell_w;
  float toxicity;
  boolean dead;
  int clogging;
  
  Person(int column, int row, int cell_w){
    this.column = column;
    this.row = row;
    this.cell_w = cell_w;
    this.toxicity = 0.0;
    this.dead = false;
    this.clogging = 0;
  }
  
  
  void move(boolean up, boolean down, boolean right, boolean left){
     if (up ){
       this.row--;
     }
     if (down){
       this.row++;
     }
     if (right){
       this.column++;
     }
     if (left){
       this.column--;
     }
     if (this.column == 0 && this.row != 20){
       println("error");
     }
  }
  

  
  void display(){
    if (this.dead){
      fill(255,0,0);
    }else{
      fill((int)(this.toxicity*2550), 0, 255 - (int)(this.toxicity*2550));
    }
    stroke(0);
    rect(this.column*this.cell_w, this.row*this.cell_w, this.cell_w, this.cell_w);
  }






}
