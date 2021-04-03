
EscapePanic ep;
static int count = 0;
static int numPeople = 1000;
static int slowness = 1;
static int end = 0;
static float toxicity = 0.03;
static boolean clogging = false;

void setup() {
  size(600, 400);
  frameRate(24);
  ep = new EscapePanic(10, numPeople, clogging, toxicity);
  ep.display();
}

void draw() {
  background(255);
  count ++;

  //gol.evaluateNextState();
  if (count%slowness == 0) {
    ep.movePeople_2();
  }
  ep.display();

  if (count%1 == 0) {
    ep.spreadSmoke();
    ep.poison();
  }
  
  if(end == 1){
    noLoop();
  }

}
