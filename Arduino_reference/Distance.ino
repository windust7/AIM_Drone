int trigPin=6;
int echoPin=7;
void setup() {
Serial.begin(9600);  //시리얼 Baudrate 9600으로 설정
pinMode(trigPin, OUTPUT); //trig를 출력모드로 설정
pinMode(echoPin, INPUT);    //echo를 입력모드로 설정
}
void loop() {
float duration, distance;
digitalWrite(trigPin, HIGH);    //초음파를 보내고 대기
delay(10);  //0.01초 대기
digitalWrite(trigPin, LOW);
duration=pulseIn(echoPin, HIGH);  //echoPin이 HIGH를 유지한 시간 저장(1,000,000s^(-1)단위)
distance=((float)(340*duration)/10000)/2; //초음파를 보내고 다시 돌아온 시간을 측정해 거리 계산(1m=100cm, duration 시간 단위)
Serial.print(distance);
Serial.println(“cm”)
Delay(500);
}
