//2motortest


#define STEP_PIN_1 6
#define DIR_PIN_1 7
#define STEP_PIN_2 12
#define DIR_PIN_2 13

void setup() {
  pinMode(STEP_PIN_1, OUTPUT);
  pinMode(DIR_PIN_1, OUTPUT);
  pinMode(STEP_PIN_2, OUTPUT);
  pinMode(DIR_PIN_2, OUTPUT);
}

void loop() {
  digitalWrite(DIR_PIN_1, HIGH);
  digitalWrite(DIR_PIN_2, HIGH);

  for (int i = 0; i < 200; i++) {
    digitalWrite(STEP_PIN_1, HIGH);
    digitalWrite(STEP_PIN_2, HIGH);
    delayMicroseconds(2000);
    digitalWrite(STEP_PIN_1, LOW);
    digitalWrite(STEP_PIN_2, LOW);
    delayMicroseconds(2000);
  }

  delay(1000);

  digitalWrite(DIR_PIN_1, LOW);
  digitalWrite(DIR_PIN_2, LOW);

  for (int i = 0; i < 200; i++) {
    digitalWrite(STEP_PIN_1, HIGH);
    digitalWrite(STEP_PIN_2, HIGH);
    delayMicroseconds(2000);
    digitalWrite(STEP_PIN_1, LOW);
    digitalWrite(STEP_PIN_2, LOW);
    delayMicroseconds(2000);
  }

  delay(1000);
}