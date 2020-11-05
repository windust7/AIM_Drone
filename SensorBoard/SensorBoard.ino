#include "PPMrcIn.h"
#include "Statistic.h"

//////////////////////CONFIGURATION///////////////////////////////
#define chanel_number 1  //set the number of chanels
#define default_servo_value 1000  //set the default servo value
#define PPM_FrLen 22500  //set the PPM frame length in microseconds (1ms = 1000µs) //22500
#define PPM_PulseLen 300  //set the pulse length // 300
#define onState 0  //set polarity of the pulses: 1 is positive, 0 is negative
#define sigPin 4  //set PPM signal output pin on the arduino 

#define PPM_Pin 2 //read ppm pin
#define multiplier (F_CPU/8000000)  //leave this alone

#define GP_FRONT_SONAR_DISTANCE_OF_AVOID_WHILE_MISSION_MIN  0
#define min_distance  5
#define GP_FRONT_SONAR_DISTANCE_OF_AVOID_WHILE_MISSION_MAX  150 
#define max_distance  400
#define GP_FRONT_SONAR_AVOID_WHILE_MISSION_COUNT_LIMIT  8
#define num_of_prev_distances 1

#define trigPin 7
#define echoPin 8
#define delay_of_loop 10  // 10, do not recommend to change this
//////////////////////////////////////////////////////////////////

int ppm_read[1];  //array for storing up to signals
int ppm[chanel_number];
int sonarFrontDistanceWhileMissionCount = 0;
int prev_distances[num_of_prev_distances];

Channel channel1;

void setup()
{ 
  Serial.begin(115200);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  //initiallize default ppm values
  for (int i = 0; i < chanel_number; i++) {
    ppm[i] = default_servo_value;
  }

  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, onState);  //set the PPM signal pin to the default state (off)

  pinMode(PPM_Pin, INPUT);
  channel1.init(1,2);

  ////////////timer1/////////////////////////
  cli(); //stop interrupts
  TCCR1A = 0; // set entire TCCR1 register to 0
  TCCR1B = 0;

  OCR1A = 20;  // compare match register, change this //20
  
  TCCR1B |= (1 << WGM12);  // turn on CTC mode
  TCCR1B |= (1 << CS11);  // 8 prescaler: 0,5 microseconds at 16mhz
  TIMSK1 |= (1 << OCIE1A); // enable timer compare interrupt
  
  sei(); //allow interrupts
  //////////////////////////////////////////
}

void loop()
{  
  int duration, distance;    //기본 변수 선언
  int sum_of_distances = 0;

  cli(); //stop interrupts
  
  //Trig 핀으로 10us의 pulse 발생
  digitalWrite(trigPin, LOW);        //Trig 핀 Low
  delayMicroseconds(2);            //2us 유지
  digitalWrite(trigPin, HIGH);    //Trig 핀 High
  delayMicroseconds(10);            //10us 유지
  digitalWrite(trigPin, LOW);        //Trig 핀 Low

  //Echo 핀으로 들어오는 펄스의 시간 측정
  duration = pulseIn(echoPin, HIGH);        //pulseIn함수가 호출되고 펄스가 입력될 때까지의 시간. us단위로 값을 리턴.
  
  sei(); //allow interrupts
  
  //음파가 반사된 시간을 거리로 환산
  //음파의 속도는 340m/s 이므로 1cm를 이동하는데 약 29us.
  //따라서, 음파의 이동거리 = 왕복시간 / 1cm 이동 시간 / 2 이다.
  distance = duration / 29 / 2;        //센치미터로 환산

  if (distance > max_distance) {
    distance = max_distance;
  }

  if (distance < min_distance) {
    distance = max_distance;
  }

  //////filtering sonar error//////////////
  sum_of_distances = 0;
  for(int i = 0 ; i < sizeof(prev_distances) / sizeof(int) ; i++) {
    sum_of_distances = sum_of_distances + prev_distances[i];
    if(i > 0) {
      prev_distances[i-1] = prev_distances[i]; 
    }
  }
  distance = (distance + sum_of_distances) / ((sizeof(prev_distances) / sizeof(int)) + 1);
  prev_distances[(sizeof(prev_distances) / sizeof(int)) - 1] = distance;
  
  ///////////////////////////////////////////
  channel1.readSignal();
  
  //Serial.print(distance);
  //Serial.print("  ");
  //Serial.print(channel1.getSignal());
  //Serial.print("  ");
  //Serial.print(sonarFrontDistanceWhileMissionCount);
  //Serial.println();

  if(1800 < channel1.getSignal()) {
    if (GP_FRONT_SONAR_DISTANCE_OF_AVOID_WHILE_MISSION_MIN < distance 
    && distance < GP_FRONT_SONAR_DISTANCE_OF_AVOID_WHILE_MISSION_MAX) {
      sonarFrontDistanceWhileMissionCount++;
      if(sonarFrontDistanceWhileMissionCount > GP_FRONT_SONAR_AVOID_WHILE_MISSION_COUNT_LIMIT) {
        ppm[0] = 2300; //2300
      }
    }
    else {
      ppm[0] = 1400;
      sonarFrontDistanceWhileMissionCount = 0;
    }  
  }
  else {
    ppm[0] = 1400;  
  }
  
  // 1400 -> 1102
  // 1800 -> 1500
  // 2000 -> 1703
  // 2300 -> 2000
  // 2500 -> 2000

  delay(delay_of_loop);
}

ISR(TIMER1_COMPA_vect) { //leave this alone
  cli(); //stop interrupts
  
  static boolean state = true;

  TCNT1 = 0;

  if (state) { //start pulse
    digitalWrite(sigPin, onState);
    OCR1A = PPM_PulseLen * 2;
    state = false;
  }
  else { //end pulse and calculate when to start the next pulse
    static byte cur_chan_numb;
    static unsigned int calc_rest;

    digitalWrite(sigPin, !onState);
    state = true;

    if (cur_chan_numb >= chanel_number) {
      cur_chan_numb = 0;
      calc_rest = calc_rest + PPM_PulseLen;//
      OCR1A = (PPM_FrLen - calc_rest) * 2;
      calc_rest = 0;
    }
    else {
      OCR1A = (ppm[cur_chan_numb] - PPM_PulseLen) * 2;
      calc_rest = calc_rest + ppm[cur_chan_numb];
      cur_chan_numb++;
    }
  }
  sei(); //allow interrupts
}

void read_ppm(){  //leave this alone
  cli(); //stop interrupts

  static unsigned int pulse;
  static unsigned long counter;
  static byte channel;

  counter = TCNT2;
  TCNT2 = 0;
  
  channel = 0;
  
  if(55*multiplier < counter && counter < 57*multiplier){  // 710->71 // 55*multiplier < counter && 
    pulse = counter;
    ppm_read[channel] = (counter + pulse)/multiplier;
  }
  else {
    ppm_read[channel] = 0;
  }
  sei(); //allow interrupts
}
