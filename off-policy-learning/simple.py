from typing import List
import matplotlib.pyplot as plt


class OffPolicy:
    """7.10

        Devise a small off-policy prediction problem and use it to show
        that the off-policy learning algorithm
        using (7.13) and (7.2) is more data efficient
        than the simpler algorithm using (7.1) and (7.9).
    """
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.95
        self.rho = 0.5

    def calculate_n_step_return(self,
                                rewards: float,
                                values: List[float]) -> float:
        """7.1

        return G (n-step return)
        """
        return (self.gamma ** rewards) + (self.gamma ** values[-1])

    def update_value_7_9(self, value: float, n_step_return: float):
        """7.9

        update V (value function)
        """

        new_value = value + self.alpha * self.rho * (n_step_return - value)
        return new_value

    def calculate_n_step_return_with_rho(self,
                                         rewards: float,
                                         values: List[float]) -> float:
        """7.13

        return G (n-step return) with sampling ratio rho
        """
        rho_return = self.rho * (
            sum([self.gamma ** rewards])
            + self.gamma ** values[-1]
        )

        return rho_return + (1 - self.rho) * values[-1]

    def update_value_7_2(self, value: float, n_step_return: float) -> float:
        """7.2

        update V (value function) with sampling ratio rho
        """
        return value + self.alpha * (n_step_return - value)
    
    def draw(self, values_a: List[float], values_b: List[float]) -> None:
        plt.plot(values_a, label="7.1 + 7.9")
        plt.plot(values_b, label="7.13 + 7.2")
        plt.xlabel('Iterations')
        plt.ylabel('Updated Value')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    off_policy = OffPolicy()
    rewards_a = 0
    updated_values_a = [0]

    n = 1000
    for _ in range(n):
        # 7.1 + 7.9
        n_step_return_19 = off_policy.calculate_n_step_return(
            rewards_a, updated_values_a)
        updated_value_19 = off_policy.update_value_7_9(
            updated_values_a[-1], n_step_return_19)
        updated_values_a.append(updated_value_19)

    print(updated_values_a)

    rewards_b = 0
    updated_values_b = [0.1]
    for _ in range(n):
        # 7.13 + 7.2 (with rho)
        n_step_return_132 = off_policy.calculate_n_step_return_with_rho(
            rewards_b, updated_values_a)
        updated_value_132 = off_policy.update_value_7_2(
            updated_values_b[-1], n_step_return_132)
        updated_values_b.append(updated_value_132)

    print(updated_values_b)

    off_policy.draw(updated_values_a, updated_values_b)
