// 아두이노 연결된 핀 설정
#define ENA 10
#define EN1 9
#define EN2 8
#define Looptime 50
#define trigPin 6
#define echoPin 7
float desired_distance=0;
float err_B=0;
float Kp=0.5;
float Ki=0.5;
float Kd=1;
float P_Control;
float I_Control;
float D_Control;
float PID_Control;
float error=0;
float error_previous=0;
float duration, distance;
float boundary=1;
unsigned long t_now;
unsigned long t_prev;
int go=1;


void setup()
{
// PWM 제어핀 출력 설정
pinMode(ENA,OUTPUT);
// 방향 제어핀 출력 설정
pinMode(EN1,OUTPUT);
pinMode(EN2,OUTPUT);
// 시리얼모니터 사용
Serial.begin(9600);
pinMode(trigPin, OUTPUT);  
pinMode(echoPin, INPUT);    
}

void loop()
{
  desired_distance=Serial.parseFloat();
  if (desired_distance!=0)
  {
    t_prev=millis();
    int d=1;
    while(d)
    {
      t_now=millis();
      if (t_now-t_prev>Looptime)
      {
        Get_Position_Decide();
        Calculate_PID();
        Run_Motor();
        t_prev=t_now;
        Get_Position_Decide();
        if (!go)
        {d=0;
        Print_Result();}
      }
    }
  }
}

void Get_Position_Decide()
{
  digitalWrite(trigPin, HIGH);
  delay(10);
  digitalWrite(trigPin, LOW);
  duration=pulseIn(echoPin, HIGH);
  distance=((float)(340*duration)/10000)/2;

  if (abs(distance-desired_distance)<abs(boundary))
  {
    go=0;
  }
}

void Calculate_PID()
{
  error=distance-desired_distance;
  P_Control=Kp*error;
  I_Control+=Ki*error*Looptime;
  float error_deriv=error-error_previous;
  if (abs(error_deriv)>0.1)
  {
    if ((error_deriv)<0)
    {
      (error_deriv)=-0.05;
    }
    else 
    {
      (error_deriv)=0.05;
    }
  }
  D_Control=Kd*(error_deriv)/Looptime;
  PID_Control=P_Control+I_Control+D_Control;
  error_previous=error;
}

void Run_Motor()
{
  if (go)
  {
    if (PID_Control>0)
    {
      digitalWrite(EN1, LOW);
      digitalWrite(EN2, HIGH);
      analogWrite(ENA, PID_Control);
      delay(30);
      analogWrite(ENA, 0);
    }
    else
    {
      digitalWrite(EN1, HIGH);
      digitalWrite(EN2, LOW);
      analogWrite(ENA, -(PID_Control));
      delay(30);
      analogWrite(ENA, 0);
    }
  }
}

void Print_Result()
{
  
  Serial.print("Desired Distance= ");
  Serial.print(desired_distance);
  Serial.print("\n");
  Serial.print("Distance= ");
  Serial.print(distance);
  Serial.print("\n");
}
