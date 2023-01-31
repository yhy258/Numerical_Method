# Numerical_Method

Jax, Numpy 기반 코딩 연습.
..  
  
### 2023 01 29.  
GMM이나 이런 method는 크기와 dimension이 큰 dataset에 대해 연산 시간을 크게 줄일 수 있는 jax가 유효하지만, numerical method 구현은 보통 one dimension dataset에 대해 진행하고 iterative한 method가 굉장히 많아서 jax를 사용할 이유가 없다.  

### 2023 01 31.

**Quadratic Splines**
-> 풀어야하는 전체 파라미터의 개수 : 3 * (n-1)
1. 주어진 interval의 시작점과 끝점에 대한 x, y 값이 일치한다. -> 2n- 2개의 equation이 풀린다.
2. 인접한 interval 사이의 미분값이 동일하다. -> n-2개의 equation이 풀린다.
3. 시작점 혹은 끝점의 이차 미분의 값이 동일하다. -> 1개의 equation이 풀린다.
  
-> python 상에서 이렇게 따로노는 condition을 처리하기가 까다로움. 구현 제외.
