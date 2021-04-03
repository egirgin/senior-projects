// The Nature of Code
// Daniel Shiffman
// http://natureofcode.com

class EscapePanic {

  ArrayList<Person> people;
  ArrayList<Smoke> smoke;

  int cell_w;
  int columns, rows;
  int numPeople;
  int door_column = 0;
  int door_row = 20;
  float toxicity = 0.001;
  int count = 0;
  boolean clogging;
  int counter = 0;


  // Game of life board
  int[][] board;
  int[][] people_board;
  int[][] smoke_board;


  EscapePanic(int c_width, int numPeople, boolean clogging, float toxicity) {
    // Initialize rows, columns and set-up arrays
    cell_w = c_width;
    columns = width/cell_w;
    rows = height/cell_w;
    board = new int[columns][rows];
    people_board = new int[columns][rows];
    smoke_board = new int[columns][rows];
    this.numPeople = numPeople;
    this.clogging = clogging;
    this.toxicity = toxicity;

    this.people = new ArrayList();
    this.smoke = new ArrayList();

    initTable();
    initPeople();
    initGas(rows-2, columns-2);
  }

  void initTable() {
    for (int i = 0; i < columns; i++) {
      board[i][0] = 1;
      board[i][rows-1] = 1;
    }
    for (int i = 0; i < rows; i++) {
      board[door_column][i] = 1;
      //board[10][i] = 1;
      board[columns-1][i] = 1;
    }

    board[door_column][door_row] = 0;
    //board[10][door_row] = 0;
  }

  void initPeople() {
    for (int i = 0; i < this.numPeople; i++ ) {
      int x = (int) random(door_column+1, columns-1);
      int y = (int) random(1, rows-1);      
      people.add(new Person(x, y, this.cell_w));
      people_board[x][y] = 1;
    }
  }

  void initGas(int row, int column) {
    smoke.add(new Smoke(column, row, cell_w));
    smoke_board[column][row] = 1;
  }

  void movePeople() {

    for (int i = 0; i<people.size(); i++) {
      if (people.get(i).dead) {
        continue;
      }

      if (people.get(i).column == door_column && people.get(i).row == door_row) {
        people_board[people.get(i).column][people.get(i).row] = 0;
        people.remove(i);
        i--;
      } else {

        boolean up = false;
        boolean down = false;
        boolean right = false;
        boolean left = false;

        if (people.get(i).column > door_column && people_board[people.get(i).column-1][people.get(i).row] == 0 && board[people.get(i).column-1][people.get(i).row] == 0) {
          left = true;
        }
        if (people.get(i).column < door_column && people_board[people.get(i).column+1][people.get(i).row] == 0 && board[people.get(i).column+1][people.get(i).row] == 0) {
          right = true;
        }
        if (people.get(i).row > door_row && people_board[people.get(i).column][people.get(i).row-1] == 0 && board[people.get(i).column][people.get(i).row-1] == 0) {
          up = true;
        }
        if (people.get(i).row < door_row && people_board[people.get(i).column][people.get(i).row+1] == 0 && board[people.get(i).column][people.get(i).row+1] == 0) {
          down = true;
        }
        int old_column = people.get(i).column;
        int old_row = people.get(i).row;
        people_board[people.get(i).column][people.get(i).row] = 0;
        people.get(i).move(up, down, right, left);
        people_board[people.get(i).column][people.get(i).row] = 1;
        int new_column = people.get(i).column;
        int new_row = people.get(i).row;
        /*
        if (old_column == new_column && old_row == new_row) {
         
         if (people_board[people.get(i).column][people.get(i).row-1] == 0 && board[people.get(i).column][people.get(i).row-1] == 0) {
         up = true;
         }
         if (people_board[people.get(i).column][people.get(i).row+1] == 0 && board[people.get(i).column][people.get(i).row+1] == 0) {
         down = true;
         }
         
         float threshold = random(1);
         if (threshold < 0.3) {
         up = false;
         } else if (threshold < 0.6) {
         down = false;
         } else {
         up = false;
         down = false;
         }
         people_board[people.get(i).column][people.get(i).row] = 0;
         people.get(i).move(up, down, false, false);
         people_board[people.get(i).column][people.get(i).row] = 1;
         }
         */
      }
    }
  }

  void movePeople_2() {
    int dont_move_ppl = 0;
    for (int i=0; i< people.size(); i++) {

      if (people.get(i).dead) {
        continue;
      }

      if (people.get(i).column == door_column && people.get(i).row == door_row) {
        people_board[people.get(i).column][people.get(i).row] = 0;
        people.remove(i);
        i--;
      } else {
        if (clogging) {
          people.get(i).clogging = 0;
          if (people_board[people.get(i).column-1][people.get(i).row] == 1) {
            people.get(i).clogging++;
          }
          if (people_board[people.get(i).column+1][people.get(i).row] == 1) {
            people.get(i).clogging++;
          }
          if (people_board[people.get(i).column][people.get(i).row-1] == 1) {
            people.get(i).clogging++;
          }
          if (people_board[people.get(i).column][people.get(i).row+1] == 1) {
            people.get(i).clogging++;
          }

          float move_prob = random(1);

          if (people.get(i).clogging*0.25 > move_prob ) {
            dont_move_ppl++;
            continue;
          }
        }


        float min_dist = 10000;
        int min_column = 0;
        int min_row = 0;
        for ( int q = 0; q < columns; q++) {
          for ( int j = 0; j < rows; j++) {
            if (board[q][j] == 0 && people_board[q][j] == 0) {
              if (dist2door(q, j) < min_dist) {
                min_dist = dist2door(q, j);
                min_column = q;
                min_row = j;
              }
            }
          }
        }

        boolean up = false;
        boolean down = false;
        boolean right = false;
        boolean left = false;

        if (people.get(i).column > min_column && people_board[people.get(i).column-1][people.get(i).row] == 0 && board[people.get(i).column-1][people.get(i).row] == 0) {
          left = true;
        }
        if (people.get(i).column < min_column && people_board[people.get(i).column+1][people.get(i).row] == 0 && board[people.get(i).column+1][people.get(i).row] == 0) {
          right = true;
        }
        if (people.get(i).row > min_row && people_board[people.get(i).column][people.get(i).row-1] == 0 && board[people.get(i).column][people.get(i).row-1] == 0) {
          up = true;
        }
        if (people.get(i).row < min_row && people_board[people.get(i).column][people.get(i).row+1] == 0 && board[people.get(i).column][people.get(i).row+1] == 0) {
          down = true;
        }

        int old_column = people.get(i).column;
        int old_row = people.get(i).row;
        people_board[people.get(i).column][people.get(i).row] = 0;
        people.get(i).move(up, down, right, left);
        people_board[people.get(i).column][people.get(i).row] = 1;
        int new_column = people.get(i).column;
        int new_row = people.get(i).row;

        if (old_column == new_column && old_row == new_row) {
          if (people_board[people.get(i).column-1][people.get(i).row] == 0 && board[people.get(i).column-1][people.get(i).row] == 0) {
            left = true;
          }
          if (people_board[people.get(i).column+1][people.get(i).row] == 0 && board[people.get(i).column+1][people.get(i).row] == 0) {
            right = true;
          }
          if (people_board[people.get(i).column][people.get(i).row-1] == 0 && board[people.get(i).column][people.get(i).row-1] == 0) {
            up = true;
          }
          if (people_board[people.get(i).column][people.get(i).row+1] == 0 && board[people.get(i).column][people.get(i).row+1] == 0) {
            down = true;
          }

          if (up && down) {
            float threshold = random(1);

            if (threshold < 0.5) {
              up = false;
            } else {
              down = false;
            }
          }

          if (right && left) {
            float threshold = random(1);

            if (threshold < 0.5) {
              right = false;
            } else {
              left = false;
            }
          }

          people_board[people.get(i).column][people.get(i).row] = 0;
          people.get(i).move(up, down, right, left);
          people_board[people.get(i).column][people.get(i).row] = 1;
        }
      }
    }
    //print("Dont move PPL");
    //println(dont_move_ppl);
  }

  float dist2door(int column, int row) {
    return sqrt(sq(column - door_column) + sq (row - door_row));
  }




  void spreadSmoke() {

    ArrayList<Smoke> new_smoke = new ArrayList();
    for (int i = 0; i< smoke.size(); i++) {
      if (smoke.get(i).column == 0 || smoke.get(i).row == 0) { // do not move at the door
        continue;
      }
      if (board[smoke.get(i).column-1][smoke.get(i).row] == 0 && smoke_board[smoke.get(i).column-1][smoke.get(i).row] == 0) {
        new_smoke.add(new Smoke(smoke.get(i).column-1, smoke.get(i).row, cell_w));
        smoke_board[smoke.get(i).column-1][smoke.get(i).row] = 1;
      }
      if (board[smoke.get(i).column+1][smoke.get(i).row] == 0 && smoke_board[smoke.get(i).column+1][smoke.get(i).row] == 0) {
        new_smoke.add(new Smoke(smoke.get(i).column+1, smoke.get(i).row, cell_w));
        smoke_board[smoke.get(i).column+1][smoke.get(i).row] = 1;
      }
      if (board[smoke.get(i).column][smoke.get(i).row-1] == 0 && smoke_board[smoke.get(i).column][smoke.get(i).row-1] == 0) {
        new_smoke.add(new Smoke(smoke.get(i).column, smoke.get(i).row-1, cell_w));
        smoke_board[smoke.get(i).column][smoke.get(i).row-1] = 1;
      }
      if (board[smoke.get(i).column][smoke.get(i).row+1] == 0 && smoke_board[smoke.get(i).column][smoke.get(i).row+1] == 0) {
        new_smoke.add(new Smoke(smoke.get(i).column, smoke.get(i).row+1, cell_w));
        smoke_board[smoke.get(i).column][smoke.get(i).row+1] = 1;
      }
    }



    smoke = new_smoke;
    //println(smoke.size());
  }

  void poison() {
    for (Person p : people) {
      if (!p.dead && smoke_board[p.column][p.row] == 1) {
        p.toxicity += this.toxicity;
        
        float is_alive = random(1);

        if (is_alive < p.toxicity) {
          
          println((p.toxicity*255));
          counter++;
          p.dead = true;
        }

      }
    }
    int dead_ppl = 0;
    for (Person p : people) {
      if (p.dead)
        dead_ppl++;
    }

    if (dead_ppl == people.size()) {
      //print("Velocity: ");
      //println(100/project4.slowness);
      //print("Num People: ");
      //println(this.numPeople);
      //print("Dead Ratio: ");
      println(dead_ppl*100/project4.numPeople);
      //print("Time: ");
      //println(project4.count);
      //print("Toxicity: ");
      println(this.toxicity);
      //print("Counter: ");
      //println(counter);
      println("----------------------------------");
      project4.end = 1;
    }
  }


  // This is the easy part, just draw the cells, fill 255 for '1', fill 0 for '0'
  void display() {
    for ( int i = 0; i < columns; i++) {
      for ( int j = 0; j < rows; j++) {
        if ((board[i][j] == 1)) fill(0);
        else fill(255); 
        stroke(0);
        rect(i*cell_w, j*cell_w, cell_w, cell_w);
      }
    }
  
    for ( int i = 0; i < columns; i++) {
      for ( int j = 0; j < rows; j++) {
        if ((smoke_board[i][j] == 1)) { 
          fill(69, 214, 44);
          stroke(0);
          rect(i*cell_w, j*cell_w, cell_w, cell_w);
        }
      }
    }


    for (int i= 0; i<people.size(); i++) {
      people.get(i).display();
    }
  }
}
