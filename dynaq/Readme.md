# Dyna-Q with pseudocode
1. Action selection: 현재 `state`를 바탕으로 할 `action`을 선택합니다. 주로 `e-greedy` 방법을 사용하여 무작위 exploration과 exploitation 사이를 조절합니다.

2. Direct RL: 다음 `state`를 관찰하고, 받은 `reward`를 이용하여 `action values`를 업데이트합니다. 이 과정에서 일반적으로 한 단계의 `tabular Q-learning`을 사용합니다.

3. Model learning: 다음 `state`와 `reward`를 이용하여 모델을 업데이트합니다. 주로 환경이 결정적(deterministic)이라고 가정하고, 테이블을 업데이트하여 모델을 유지합니다.

4. Planning: 초기 `state`와 `acition`을 이용하여 `n`개의 시뮬레이션 경험을 생성하여 `Action values`를 업데이트합니다. 주로 무작위 샘플링을 통한 한 단계의 `tabular Q-planning`를 사용합니다. 이것은 `Indirect RL`으로도 알려져 있습니다. 경험을 시뮬레이션하기 위해 `state`와 `action`을 선택하는 과정을 `search control`라고 합니다.