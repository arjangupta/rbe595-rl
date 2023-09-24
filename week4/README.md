To install required packages:
pip3 install numpy
pip3 install matplotlib

To run:
python3 dp_exercise.py <model type> <algorithm type>

Valid Values for Model Type:
- "Deterministic"
  - always executes movements perfectly
- "Stochastic"
  - has a 20% probability of moving +/-45degrees from the commanded move

Valid Values for Algorithm Type:
- "PolicyIteration"
  - algorithm on page 80 Sutton and Barto
- "ValueIteration"
  - algorithm on page 83 Sutton and Barto
- "GeneralizedPolicyIteration"
  - similar to policy iteration, except the policy evaluation and policy iteration can interact independently, regardless of their convergence