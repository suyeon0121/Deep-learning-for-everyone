from tensorflow.keras.models import Sequential              # keras API에 있는 model 클래스에서 Sequential 함수 호출
from tensorflow.keras.layers import Dense                   # keras API에 있는 layers 클래스에서 Dense 함수 호출
import numpy as np                                          # 라이브러리명이 길거나 같은 이름이 있을 경우 numpy 라이브러리를 np라는 짧은 이름으로 호출

Data_set = np.loadtxt('./data/ThoraricSurgery3.csv', delimiter=",")             # Data_set 변수 / loadtxt -> 파일 읽어오는 함수 / delimiter 구분자
X = Data_set[:,0:16]                                        # 쉼표 기준으로 앞은 행, 뒤는 열 -> 모든 행부터 1번째 열부터 16번째 열까지 / 상수(대문자)
y = Data_set[:,16]                                          # 17번째 값을 추출 / 변수(소문자)   

model = Sequential()                                        # 입력층 : Sequential()함수를 model로 선언
model.add(Dense(30, input_dim=16, activation='relu'))       # 은닉층 : Dense(30의 노드), input_dim(폐암환자 생존여부 데이터 17개를 노드로 보냄), activation(활성화 함수)='relu'(기울기소실v, 속도 ^)
model.add(Dense(1, activation='sigmoid'))                   # 출력층 : 노드 1개 -> 출력 값 1개(0, 1), sigmoid 활성화 함수 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # loss : 손실함수 , optimizer : 최적화 방법, metrics : 정확도
history = model.fit(X, y, epochs=5, batch_size=16)