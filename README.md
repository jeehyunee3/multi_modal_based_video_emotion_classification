# Multi-Modal based ViT Model for Video Data Emotion Classification


> **영상 데이터 감정 분류를 위한 멀티 모달 기반의 ViT 모델**<br/>
> 본 연구에서는 사전 학습된 VGG(Visual Geometry Group)모델과 ViT(Vision Transformer) 모델을 이용하여<br/>
> 영상 데이터에서 오디오와 이미지를 추출해 멀티 모달 방식의 영상 감정 분류 모델을 제안한다.

---
<br/>

### **1. ViT 모델과 CNN 기반 모델과의 비교 + ViViT**
 ViT  모델을 통하여 동영상의 emotion을 분류하고 CNN 기반 모델과 성능을 비교한다.  
 이때, 대조를 위한 CNN 모델로는 현재 vision task SOTA 모델인 **VGG를 사용**한다.
### **2. 멀티 모달 모델(레이어) 병합 방식 탐색**
하나의 비디오에서 추출한 **두 개의 데이터 (이미지, 오디오)** 를 사용하여 학습할 때, <br/> **모델, 레이어의 병합** 과정에 있어 어떠한 병합 방식이 가장 우수한 결과를 도출하는지 확인한다.

`영상 콘텐츠(Video content)` `감정분류(Emotion classification)` `VGG(Visual Geometry Group)`<br/> `ViT(Vision Transformer)` `ViViT(Video Vision Transformer)` 

<br/>
<br/>

## Project Information
<br/>
 - **프로젝트 기간**
   	 - 1차 : 2022.10.12 ~ 2023.01.13
	 - 2차 : 진행중 
	 
 - **프로젝트 참여자** 
   	- kimyerim0908@gmail.com
   	- 98_0731@naver.com
   	- jeehyunee3@naver.com
   	-  220216138@sungshin.ac.kr
   	<br/>


<br/>

## Model Architecture
### 전체적 모델 구조
![enter image description here](https://user-images.githubusercontent.com/72274498/237034674-c1855890-b029-4a7e-b629-67cd535d42d9.png)

<br/>

### 모델(레이어) 병합 방식에 따른 2가지 접근 방식
![enter image description here](https://user-images.githubusercontent.com/72274498/237036687-90369a1c-7106-4f99-ac5e-ec010ecdd73c.png)

<br/>

## Paper

[영상 데이터 감정 분류를 위한 멀티 모달 기반의 ViT 모델 - 한국컴퓨터정보학회 학술발표논문집 - 한국컴퓨터정보학회 : 논문 - DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11213359)
- 한국컴퓨터정보학회 
- 2023년 한국컴퓨터정보학회 동계학술대회 논문집 제 31권 1호, pg. 9-12, 2023.01
- 김예림, 이동규, 안서영, 김지현
- 성신여자대학교

<br/>

## Data
본 연구에서는 **AI HUB**에서 제공하는 **### 동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터** 데이터 셋을 이용하였습니다. <br/>
[AI-Hub (aihub.or.kr)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=616)

![enter image description here](https://user-images.githubusercontent.com/72274498/237035348-18d05002-1ed7-440d-9578-3fecdf767f1c.png)


<br/>

## Performances
![enter image description here](https://user-images.githubusercontent.com/72274498/237037705-514c58d7-9aac-4a4f-a9ee-b319bccf659f.png)


