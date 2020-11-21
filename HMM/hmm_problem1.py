import numpy as np
from hmmlearn import hmm
states = ["box1", "box2", "box3"]
n_states = len(states)
observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])
transition_probabiliy= np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.5]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probabiliy
model.emissionprob_ = emission_probability

seen = np.array([[1, 0, 0]]).T
box = model.predict(seen)
print(box)

score = model.score(seen)
print(score)
# print("The ball picked:", ",".join(map(lambda x: observations[x], seen)))
# print("the hidden box", ",".join(map(lambda x: states[x], box)))
# print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
# print("The hidden box", ", ".join(map(lambda x: states[x], box)))