"""
1. 결과가 0 혹은 1이고 E(P(X))를 조절할 수 있는 arm을 2개 (A, B) 만드세요. 10%로 설정하면 10%의 확률로 1이 나와야 하는 arm입니다.
2. greedy, e-greedy, UCB, thompson sampling을 구현하세요. 하이퍼파라미터는 적당하게 설정하고 바꾸지 않습니다.
3. A의 승률이 10%, 20%, …, 90%, B의 승률이 10%, 20%, …, 90%이고 위 arm에 대해서 5번씩 테스트하여 각 알고리즘별 평균 승수를 구하세요. 총 9 X 9 X 5 X 4번 테스트해야 하고 최종 결과는 9 X 9 X 4 만큼 나와야 합니다.
4. 위 결과를 scatter plot으로 표현하세요. x축은 E(P(A))가 10%, 20%, …, 90% 이고 y축은 E(P(B))가 10%, 20%, …, 90%를 의미합니다. color는 알고리즘, size는 알고리즘별 평균 승수입니다.
5. 코드는 각자의 깃헙에, 최종 이미지 1장은 이 채널에 제출합니다.
"""

import random


class Arm:
    def __init__(self, probability):
        self.probability = probability

    def pull(self) -> int:
        if random.random() < self.probability:
            return 1

        return 0


arm_a = Arm(0.1)
arm_b = Arm(0.1)

time_step = 100
total_rewards = [0, 0]

for _ in range(time_step):
    p_a = arm_a.pull()
    p_b = arm_b.pull()

    total_rewards[0] += p_a
    total_rewards[1] += p_b

print(total_rewards)
