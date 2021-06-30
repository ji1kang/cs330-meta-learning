# Lecture Note 1 - Introduction & Overview

- 강의영상: [Lecture Note 1 - Introduction & Overview](https://youtu.be/0rZtSwNOTQo)

### 메타러닝이란?
- 메타러닝은 다음의 문제를 해결하기 위해 사용할 수 있습니다.
	- What if you need to quickly learn something new?
	- What if you don’t have a large dataset?
- 이러한 문제를 해결하기 위해 모델이 학습한 이전의 경험들을 사용하고 조정합니다(by leveraging prior experience!).

### Critical Assumption: 멀티테스크 러닝의 좋은점과 나쁜점
- 나쁜점: 각자 다른 테스크가 모델의 구조를 공유해야할 필요가 있다. 즉, 이렇게하면 문제를 해결할 수 없는 문제면 기존의 싱글 테스트 러닝을 선택할 것.
(Different tasks need to share some structure. <- If this doesn’t hold you are better off using single-task learning)

- 좋은점: 사실 많은 테스크들이 모델의 구조를 공유할 수 있으며, 이런 점은 각자 연관없는 랜덤 테스크에 멀티테스크 러닝을 적용할때보다 더 좋은 성능을 얻을 수 있다.
(The good news: There are many tasks with shared structure! -> This leads to far greater structure than random tasks)

### Informal problem definitions: 멀티테스크 러닝 vs. 메타러닝
- 멀티테스크 러닝은 각자 학습하는 것보다 다수의 테스크를 빠르고 능숙하게 해결할 수 있음.
(The multi-task learning problem: Learn all of the tasks more quickly or more proficiently than learning them independently.)
- 메타러닝은 이전에 학습한 데이터와 경험을 이용해서, 새로운 테스크를 더 빠르고 능숙하게 해결할 수 있음.
(The meta-learning problem: Given data/experience on previous tasks, learn a new task more quickly and/or more proficiently.)
