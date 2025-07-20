# AI 기술면접 스터디 📚

데이터 사이언티스트 / AI 엔지니어 취업을 위한 체계적인 기술면접 준비 스터디입니다.

## 📅 스터디 일정

| 일자 | 주제 | 세부 내용 | 비고 |
|------|------|-----------|------|
| **2025.07.09** | **Kick-off** | 경험기술서 작성법, 실무면접의 2가지 종류<br/>- 프로젝트 vs 기본기 면접 | |
| **2025.07.16** | **머신러닝** | 회귀/분류 모델, 평가지표, Ensemble<br/>Hyperparameter Tuning, Cross-validation | |
| **2025.07.23** | **통계** | 기초통계, 확률분포, 가설검정 등 | |
| **2025.07.30** | **딥러닝 (1)** | 딥러닝 기초, MLP, Backpropagation<br/>Activation Function, Loss Function<br/>경사하강법, Optimizer, Perceptron | |
| **2025.08.06** | **딥러닝 (2)** | CNN 계열 모델, RNN 계열 모델 ~ GPT까지 | |
| **2025.08.13** | **LLM (1)** | ChatGPT, LLaMA 등 LLM 모델, sLLM 모델 | |
| **2025.08.20** | **LLM (2)** | RAG | |
| **2025.08.27** | **기술스택** | | |
| **2025.09.03** | **Wrap up** | | |

## 🎯 스터디 범위

### ✅ 포함 범위
- 머신러닝/딥러닝 기초 이론
- 통계학 기본 개념
- LLM 및 최신 AI 기술
- 기술스택 관련 지식

### ❌ 제외 범위
- 이미지, 음성, 오디오 처리
- 자료구조, 알고리즘
- 운영체제, 네트워크

### 💡 선택 학습
- 데이터베이스 (공부하면 좋음)

## 📖 참고 자료

- **머신러닝/딥러닝/통계수학/파이썬/자료구조알고리즘**: [AI-Tech-Interview](https://github.com/boost-devs/ai-tech-interview)
- **LLM**: [LLM Interview Questions](https://github.com/llmgenai/LLMInterviewQuestions)

## 📝 스터디 커리큘럼

### 1. 정형데이터 회귀 📈

#### 기본 개념
- 평가지표, Loss Function

#### 모델
- **선형 모델**: 단순선형회귀, 다중선형회귀, 다항회귀
- **정규화 모델**: 릿지회귀, 라쏘회귀, 엘라스틱넷 회귀
- **트리 기반**: 트리기반 회귀모델
- **딥러닝**: 딥러닝 회귀모델

#### Loss Function 최적화 방법
- 정규 방정식(Normal Equation)
- 특이값 분해(SVD)  
- 경사하강법(Gradient Descent)

### 2. 정형데이터 분류 📊

#### 평가지표
- **Confusion Matrix** (Scikit-learn)
- **기본 지표**: Accuracy, Error Rate, Precision, Recall (Scikit-learn)
- **복합 지표**: F1 Score, F-Beta Score (Scikit-learn)
- **확률 기반**: AUROC, AUPRC (Scikit-learn)
- **통계 기반**: KS-stat (Scikit-learn)

#### 모델
- **전통적 모델**: 로지스틱 회귀, k-NN, SVM (Scikit-learn)
- **트리 모델**: Decision Tree, Random Forest (Scikit-learn)
- **부스팅**: AdaBoost, Gradient Boosting (Scikit-learn)
- **고급 부스팅**: XGBoost, LightGBM
- **딥러닝**: 딥러닝 분류 모델 (PyTorch, TensorFlow)

### 3. 성능 개선 방법 🚀

#### 교차검증
- **기본**: Holdout, k-Fold, Stratified k-Fold (Scikit-learn)
- **특수**: LOOCV, Time Series Cross Validation (Scikit-learn)
- **반복**: Repeated K-Fold Cross Validation (Scikit-learn)

#### 하이퍼파라미터 튜닝
- **전역 탐색**: Grid Search (Scikit-learn)
- **랜덤 탐색**: Random Search (Scikit-learn)
- **베이지안 최적화**: Bayesian Search (Optuna)

#### 샘플링 & 앙상블
- **샘플링**: 언더샘플링, 오버샘플링, 부트스트래핑
- **앙상블**: Voting, Bagging, Boosting, Stacking

#### 고급 기법
- **피처 엔지니어링**: 피처 가공, 합성데이터 생성
- **모니터링**: 성능 모니터링, Drift 탐지, 성능 하락 대응

### 4. 데이터 전처리 🔧

- **이상치 처리**: 이상치 종류 및 처리 방식
- **결측치 처리**: 결측치 종류 및 처리 방식  
- **Feature Scaling**: 정규화, 표준화
- **Feature Encoding**: 범주형 데이터 인코딩

### 5. 통계학 📊

#### 기술통계
- **기술통계량**: 평균, 중앙값, 최빈값, 분산/표준편차, 사분위수, 왜도, 첨도
- **시각화**: 히스토그램, 박스플롯, 막대그래프 등

#### 추론통계
- **확률분포**: 정규분포, t분포, z분포, F분포, 베르누이분포, 이항분포, 포아송분포
- **통계적 추정**: 신뢰도, 표준오차, 모평균 추정, 모비율 추정
- **가설검정**: 
  - 기본 개념: 귀무가설/대립가설, 유의수준, p-value, 검정 오류
  - 평균값 검정: 단일표본/독립표본/대응표본 t-검정
  - 기타 검정: 비율검정, 분산검정(F-test), 상관성검정, 회귀분석

## 🎯 스터디 진행 방식

**이론 중심 접근법**을 통해 개념 이해를 우선으로 합니다.

각 주제별로 다음과 같은 구조로 학습합니다:
- **개념 설명**: 핵심 이론 정리
- **장단점 분석**: 각 방법론의 특징
- **꼬리질문 대응**: 심화 질문 준비  
- **코드 실습**: 주요 패키지 활용 (필요시)

## 🤝 스터디 운영 방안

1. **개별 학습 + 그룹 토론**: 각자 질문 답변을 준비하고 주 1회 함께 검토
2. **팀 기반 학습**: 팀을 이뤄 질문 답변을 공유하고 피드백 교환

## 📌 학습 목표

AI/데이터 분야 기술면접에서 자주 출제되는 핵심 개념들을 체계적으로 학습하여, 실무진과의 기술 토론에서 논리적이고 정확한 답변을 할 수 있도록 준비합니다.

---

*"준비된 자에게 기회는 찾아온다"* 💪