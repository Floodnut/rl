![exercise](./7.10.png)

## Exercise 7.10 (programming)

Devise a small off-policy prediction problem and use it to show that the off-policy learning algorithm using (7.13) and (7.2) is more data efficient than the simpler algorithm using (7.1) and (7.9).

### 7.9의 경우 (+ 7.1)

샘플링 비율 `ρ`를 값 함수 `V`를 업데이트하기 위해 사용한다.

- V의 업데이트에 직접적으로 관여.
- 현재 시점의 `V`를 기준으로 샘플링 비율을 적용하여 값 함수를 업데이트.

### 7.13의 경우 (+ 7.2)

샘플링 비율 `ρ`를 `n-step return`을 계산할 때 사용한다.

- `ρ`는 `n-step return`을 계산하는 과정에서 다른 정책으로부터 얻은 데이터를 현재 정책에 맞게 조정하는 데 활용.
- `n-step return`을 계산할 때 다른 정책의 데이터를 현재 정책에 적용할 때의 가중치를 나타냄.
- 샘플의 가중치를 조정하여 `n-step return`을 계산
