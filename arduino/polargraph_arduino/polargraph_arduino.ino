// ============================================
// POLARGRAPH PLOTTER - Arduino Controller
// Robot Art Dissertation Project
// ============================================
// 
// This code receives (x,y) coordinates from MATLAB
// via serial, converts them to string lengths using
// inverse kinematics, and steps the motors accordingly.
//
// Hardware:
//   - Arduino UNO R4 WiFi
//   - 2x NEMA 17 stepper motors
//   - 2x A4988 motor drivers
//   - GT2 belt + 20T pulleys (2mm pitch)
//
// Serial Protocol:
//   Send "G x y\n" to move to position (x,y) in mm
//   Send "H\n" to home (go to starting position)
//   Send "P\n" to report current position
//   Arduino replies "OK\n" when move is complete
// ============================================

// --- PIN DEFINITIONS ---
#define STEP_PIN_L  3   // Left motor step
#define DIR_PIN_L   2   // Left motor direction
#define STEP_PIN_R  5   // Right motor step
#define DIR_PIN_R   4   // Right motor direction

// --- MACHINE PARAMETERS ---
float D = 600.0;              // Distance between motor shafts in mm 
float STEPS_PER_REV = 200.0;  // NEMA 17 = 200 steps/rev (1.8° per step)
float MM_PER_REV = 40.0;      // GT2 20T pulley: 20 teeth * 2mm pitch = 40mm per rev
float STEPS_PER_MM;            // Calculated in setup()

// --- CURRENT STATE ---
float currentX;     // Current pen position X (mm)
float currentY;     // Current pen position Y (mm)
long stepsL = 0;    // Current step count for left motor
long stepsR = 0;    // Current step count for right motor

// --- SPEED CONTROL ---
int stepDelay = 800; // Microseconds between steps (lower = faster)
                      // 800us is a safe starting speed
                      // Can go down to ~400 for faster drawing

// --- SERIAL BUFFER ---
String inputBuffer = "";

void setup() {
  // Set pin modes
  pinMode(STEP_PIN_L, OUTPUT);
  pinMode(DIR_PIN_L, OUTPUT);
  pinMode(STEP_PIN_R, OUTPUT);
  pinMode(DIR_PIN_R, OUTPUT);
  
  // Calculate steps per mm
  STEPS_PER_MM = STEPS_PER_REV / MM_PER_REV;  // 200/40 = 5 steps per mm
  
  // Start at centre-top position (middle of the two motors, slightly below)
  currentX = D / 2.0;
  currentY = 100.0;  // 100mm below the motor line
  
  // Calculate initial step positions
  stepsL = lengthToSteps(calculateL1(currentX, currentY));
  stepsR = lengthToSteps(calculateL2(currentX, currentY));
  
  // Start serial
  Serial.begin(115200);
  Serial.println("POLARGRAPH READY");
  Serial.print("D = ");
  Serial.print(D);
  Serial.println(" mm");
  Serial.print("Steps/mm = ");
  Serial.println(STEPS_PER_MM);
  Serial.print("Starting position: (");
  Serial.print(currentX);
  Serial.print(", ");
  Serial.print(currentY);
  Serial.println(")");
  Serial.println("Send 'G x y' to move, 'H' to home, 'P' for position");
}

void loop() {
  // Read serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}

// ============================================
// INVERSE KINEMATICS
// ============================================
// Given a target (x, y) position on the whiteboard,
// calculate the required string lengths L1 and L2.
//
// Coordinate system:
//   Origin (0,0) = left motor position
//   X = horizontal (positive to the right)
//   Y = vertical (positive downward)
//   Right motor is at (D, 0)
//
// L1 = distance from left motor to pen
// L2 = distance from right motor to pen
// ============================================

float calculateL1(float x, float y) {
  return sqrt(x * x + y * y);
}

float calculateL2(float x, float y) {
  return sqrt((D - x) * (D - x) + y * y);
}

long lengthToSteps(float lengthMM) {
  return (long)(lengthMM * STEPS_PER_MM);
}

// ============================================
// MOTOR CONTROL
// ============================================

void stepMotorL(bool forward) {
  digitalWrite(DIR_PIN_L, forward ? HIGH : LOW);
  digitalWrite(STEP_PIN_L, HIGH);
  delayMicroseconds(2);
  digitalWrite(STEP_PIN_L, LOW);
}

void stepMotorR(bool forward) {
  digitalWrite(DIR_PIN_R, forward ? HIGH : LOW);
  digitalWrite(STEP_PIN_R, HIGH);
  delayMicroseconds(2);
  digitalWrite(STEP_PIN_R, LOW);
}

// ============================================
// MOVE TO POSITION
// ============================================
// Uses Bresenham-style line algorithm to move
// both motors simultaneously toward the target.
// This gives smooth diagonal movement instead
// of moving one motor at a time.
// ============================================

void moveTo(float targetX, float targetY) {
  // Calculate target string lengths
  float targetL1 = calculateL1(targetX, targetY);
  float targetL2 = calculateL2(targetX, targetY);
  
  // Convert to step counts
  long targetStepsL = lengthToSteps(targetL1);
  long targetStepsR = lengthToSteps(targetL2);
  
  // Calculate steps needed
  long deltaL = targetStepsL - stepsL;
  long deltaR = targetStepsR - stepsR;
  
  // Direction
  bool dirL = (deltaL >= 0);  // true = pay out string (longer)
  bool dirR = (deltaR >= 0);  // true = pay out string (longer)
  
  // Absolute steps
  long absL = abs(deltaL);
  long absR = abs(deltaR);
  
  // Bresenham line algorithm for simultaneous stepping
  long maxSteps = max(absL, absR);
  long errorL = 0;
  long errorR = 0;
  
  for (long i = 0; i < maxSteps; i++) {
    errorL += absL;
    errorR += absR;
    
    if (errorL >= maxSteps) {
      errorL -= maxSteps;
      stepMotorL(dirL);
      stepsL += dirL ? 1 : -1;
    }
    
    if (errorR >= maxSteps) {
      errorR -= maxSteps;
      stepMotorR(dirR);
      stepsR += dirR ? 1 : -1;
    }
    
    delayMicroseconds(stepDelay);
  }
  
  // Update current position
  currentX = targetX;
  currentY = targetY;
}


// COMMAND PROCESSING


void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("G") || cmd.startsWith("g")) {
    // GOTO command: "G x y"
    // Parse x and y values
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    
    if (firstSpace == -1 || secondSpace == -1) {
      Serial.println("ERROR: Format is 'G x y'");
      return;
    }
    float targetX = cmd.substring(firstSpace + 1, secondSpace).toFloat();
    float targetY = cmd.substring(secondSpace + 1).toFloat();
    
    // Safety bounds check
    if (targetX < 0 || targetX > D || targetY < 50 || targetY > 800) {
      Serial.println("ERROR: Position out of bounds");
      return;
    }
    // Move to target
    moveTo(targetX, targetY);
    Serial.println("OK");
    
  } else if (cmd.startsWith("H") || cmd.startsWith("h")) {
    // HOME command
    moveTo(D / 2.0, 100.0);
    Serial.println("HOME OK");
    
  } else if (cmd.startsWith("P") || cmd.startsWith("p")) {
    // POSITION report
    Serial.print("POS ");
    Serial.print(currentX);
    Serial.print(" ");
    Serial.println(currentY);
    
  } else if (cmd.startsWith("S") || cmd.startsWith("s")) {
    // SPEED command: "S value" (step delay in microseconds)
    int space = cmd.indexOf(' ');
    if (space != -1) {
      stepDelay = cmd.substring(space + 1).toInt();
      Serial.print("SPEED SET: ");
      Serial.println(stepDelay);
    }
    
  } else if (cmd.startsWith("D") || cmd.startsWith("d")) {
    // SET D command: "D value" (motor distance in mm)
    int space = cmd.indexOf(' ');
    if (space != -1) {
      D = cmd.substring(space + 1).toFloat();
      Serial.print("D SET: ");
      Serial.println(D);
    }
    
  } else {
    Serial.print("UNKNOWN: ");
    Serial.println(cmd);
  }
}
