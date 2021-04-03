class Smoke{
  int row,column,cell_w;
  
  Smoke(int column, int row, int cell_w){
    this.row = row;
    this.column = column;
    this.cell_w = cell_w;
  }
  
  void display(){
    fill(69,214,44);
    stroke(0);
    rect(this.column*this.cell_w, this.row*this.cell_w, this.cell_w, this.cell_w);
  }
  
}
