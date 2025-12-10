# 시편 마스크 표면·엣지 결함 검출 및 각도·길이 정밀 계측 (Specimen Mask Defect Detection & Precision Measurement)

> **2025-2학기 스마트팩토리 캡스톤디자인2 - 8조** 
> 딥러닝 기반 결함 스크리닝과 전통적 컴퓨터 비전 기반의 정밀 기하 측정을 결합한 하이브리드 파이프라인
>
> ![8조 포스터](https://github.com/user-attachments/assets/d0bc1eed-bfa6-4903-b968-3ea21b55f15b)

## 📋 프로젝트 개요 (Overview)

본 프로젝트는 반도체/디스플레이 제조 공정에서 사용되는 시편 마스크의 수동 검사 한계를 극복하고, 다품종 소량 생산 환경에서의 검사 시간을 단축하기 위해 개발되었습니다.
딥러닝 모델을 활용해 결함을 1차적으로 스크리닝하고, 결함이 없는 양품에 한해 컴퓨터 비전 알고리즘으로 정밀한 길이와 각도를 측정하는 **2단계 검사 파이프라인**을 제안합니다.

## 👥 팀 구성 (Team Members)
* **소속**: 성균관대학교 (SKKU) / ICT명품인재양성사업단
* **지도교수**: 정종필 교수님 
* **팀원 (8조)**: 허영호, 최성지, 변지환

## 🛠️ 시스템 아키텍처 (System Architecture)


본 시스템은 다음과 같은 두 단계 파이프라인으로 구성됩니다. 

### **Stage 1: AI 기반 결함 스크리닝 (AI-based Defect Screening)**
입력된 현미경 이미지를 분석하여 결함 유무를 판정하고 분류합니다.
* **Process**: Input Image → AI Inference (Roboflow) → Object Detection → 결함 판정
* **Defect Classes**:
    * **Chipping**: 모서리 금형 깨짐 
    * **Nick**: 국소 찍힘 
    * **Scratch**: 표면 스크래치 
* **Logic**: 결함이 검출되면(NG) 즉시 분류하고, 결함이 없는(Clean) 이미지만 Stage 2로 전달합니다.

### **Stage 2: 정밀 길이/각도 측정 (Precision Geometry Measurement)**
결함이 없는 이미지에 대해 정밀 계측을 수행합니다.
* **Tech Stack**: OpenCV, NumPy, SciPy 
* **Process**:
    1.  **Scale Bar Est.**: 스케일 바 추정 및 픽셀-µm 변환
    2.  **Segmentation**: 이미지 분할 및 전처리 
    3.  **Corner Loc.**: 네 코너(LT, LB, RT, RB) 위치 추정 
    4.  **Final Output**: 엣지 벡터 계산을 통한 길이(Length, µm) 및 각도(Angle, °) 산출 

## 📊 실험 및 성능 (Experiments & Results)


### **1. 데이터셋 (Dataset)**
* **Total Images**: 450장 (Train 60% / Val 20% / Test 20%) 
* **Test Set**: Clean 50장, Defective 40장 

### **2. 결함 검출 성능 (Defect Detection Performance)**
세 가지 모델(Roboflow 3.0, YOLOv11, RF-DETR-Seg)을 비교 실험한 결과, **RF-DETR-Seg**가 가장 우수한 성능을 보였습니다.

| Model | mAP | F1-Score | 비고 |
|:---:|:---:|:---:|:---|
| Roboflow 3.0 | 0.63 | 0.73 | |
| YOLOv11 | 0.69 | 0.77 | |
| **RF-DETR-Seg** | **0.76** | **0.83** | Best Performance |

* **Result Analysis**: Chipping 클래스에서 가장 높은 성능을 보였으며, Nick과 Scratch는 데이터 불균형으로 인해 상대적으로 낮은 성능을 기록했습니다. 임계값(Confidence Threshold) 0.5에서 재현율 0.93, 정밀도 0.88을 달성했습니다.

### **3. 정밀 계측 성능 (Measurement Accuracy)**
* **길이 측정 (Length)**: 평균 절대 오차(MAE) **0.84 µm**, 상대 오차(RE) **1.4%** (설계 길이 60-80 µm 기준) 
* **각도 측정 (Angle)**: 4개 모서리(LT, LB, RT, RB) 모두 측정 오차 **1도(1°)** 내외

## 📝 결론 및 향후 과제 (Conclusion & Future Work)

### **핵심 성과**
* 딥러닝(결함 스크리닝)과 전통적 기하 측정(CV)을 결합하여 µm 수준의 길이 오차와 1° 수준의 각도 오차를 달성했습니다.

### **한계점 (Limitations)**
* 특정 이미지 촬영 셋업에 특화되어 있어 범용성이 부족할 수 있습니다.
* Nick/Scratch와 같은 미세 결함에 대한 검출 성능이 제한적입니다.

### **향후 연구 방향 (Future Work)**
* ROI 및 임계값 자동 설정 기능 추가.
* 복잡한 형상 측정에 대한 일반화 및 불확실성 추정 통합.

