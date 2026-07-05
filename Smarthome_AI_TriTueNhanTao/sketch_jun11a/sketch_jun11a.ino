#include <Servo.h>
#include <LiquidCrystal.h>

// 1. MÀN HÌNH LCD
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// 2. KHAI BÁO CHÂN THIẾT BỊ
const int acPin    = 7;  
const int lampPin  = 8;  
const int fanPin   = 9;  
const int servoPin = 10; 
const int ledGreen = A1; 
const int ledRed   = A2; 

// CHÂN CẮM CẢM BIẾN NHIỆT ĐỘ LM35 (Cắm vào chân Analog A0)
const int lm35Pin  = A0; 

Servo blindsServo;

// Biến lưu nhiệt độ
unsigned long lastRead = 0;
float currentTemp = 25.0; 

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(20); 

  lcd.begin(16, 2);
  pinMode(acPin, OUTPUT);
  pinMode(lampPin, OUTPUT);
  pinMode(fanPin, OUTPUT);
  pinMode(servoPin, OUTPUT);
  pinMode(ledGreen, OUTPUT);
  pinMode(ledRed, OUTPUT);

  blindsServo.attach(servoPin);
  turnOffAll();
}

void turnOffAll() {
  digitalWrite(lampPin, LOW);
  digitalWrite(fanPin, LOW);
  digitalWrite(acPin, LOW);
  blindsServo.write(0); 
  digitalWrite(ledGreen, LOW);
  digitalWrite(ledRed, HIGH); 
  updateLCD("SYSTEM LOCKED", "All Devices OFF");
}

void updateLCD(String line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

void loop() {
  // ==============================================================
  // 1. ĐỌC CẢM BIẾN LM35 MỖI 1 GIÂY (MƯỢT 100%, KHÔNG LAG)
  // ==============================================================
  if (millis() - lastRead > 1000) {
    int analogVal = analogRead(lm35Pin);
    
    // Công thức toán học chuyển đổi điện áp thành độ C cho LM35
    float t = analogVal * (5.0 / 1023.0) * 100.0;
    
    currentTemp = t; 
    lastRead = millis();
  }

  // ==============================================================
  // 2. NHẬN LỆNH TỪ PYTHON VÀ ĐIỀU KHIỂN THIẾT BỊ
  // ==============================================================
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg != "0" && msg != "R") {
      digitalWrite(ledRed, LOW);
      digitalWrite(ledGreen, HIGH);
    }

    if (msg == "1" || msg == "LAMP_ON") {
      digitalWrite(lampPin, HIGH);
      updateLCD("SMART CONTROL", "Lamp: ON");
    }
    else if (msg == "LAMP_OFF") {
      digitalWrite(lampPin, LOW);
      updateLCD("SMART CONTROL", "Lamp: OFF");
    }
    else if (msg == "2" || msg == "FAN_ON") {
      digitalWrite(fanPin, HIGH);
      updateLCD("SMART CONTROL", "Fan: ON");
    }
    else if (msg == "FAN_OFF") {
      digitalWrite(fanPin, LOW);
      updateLCD("SMART CONTROL", "Fan: OFF");
    }
    else if (msg == "W") { 
      blindsServo.write(90);
      updateLCD("HAND GESTURE", "Blinds: OPEN");
    }
    else if (msg == "L") { 
      blindsServo.write(0);
      updateLCD("HAND GESTURE", "Blinds: CLOSED");
    }
    else if (msg == "AC_ON") {
      digitalWrite(acPin, HIGH);
      updateLCD("SMART HUB", "AC: ON");
    }
    else if (msg == "AC_OFF") {
      digitalWrite(acPin, LOW);
      updateLCD("SMART HUB", "AC: OFF");
    }
    else if (msg == "0") {
      turnOffAll();
    }
    
    // --- KHI PYTHON ĐÒI NHIỆT ĐỘ ĐỂ BỚM CHO AI ---
    else if (msg == "R") {
      Serial.print("TEMP_IN:");
      Serial.println(currentTemp); // Trả về nhiệt độ đo từ LM35
    }
    
    // --- KHI PYTHON RA LỆNH ĐỔI SỐ TRÊN LCD ---
    else if (msg.startsWith("SET:")) {
      if (msg.indexOf("OFF") > 0) {
        digitalWrite(acPin, LOW);
        updateLCD("SMART AC (AI)", "AC: OFF");
      } else {
        digitalWrite(acPin, HIGH);
        int firstColon = msg.indexOf(':');
        int secondColon = msg.indexOf(':', firstColon + 1);
        String tempValue = msg.substring(firstColon + 1, secondColon);
        updateLCD("SMART AC (AI)", "Target: " + tempValue + "C");
      }
    }
  }
}