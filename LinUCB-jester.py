import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181, shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
       file_name: Route of file containing the modified Jester dataset.
       context_dim: Context dimension (i.e. vector with some ratings from a user).
       num_actions: Number of actions (number of joke ratings to predict).
       num_contexts: Number of contexts to sample.
       shuffle_rows: If True, rows from original dataset are shuffled.
       shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
       dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
       opt_vals: Vector of deterministic optimal (reward, action) for each context."""
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
       dataset = np.load(f)
    if shuffle_cols:
       dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
       np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset, opt_rewards, opt_actions

"load data"
data_joke_scores, data_opt_rewards, _ = sample_jester_data('jester_data_40jokes_19181users.npy')

"parameters, initialize"
alpha = 1.0
num_arms = 8
d = 32

A = np.zeros((num_arms, d, d))
for a in range(num_arms):
    A[a, :, :] = np.eye(d)   
b = np.zeros((num_arms, d))
ucbs = np.zeros(num_arms)

first_n_train_rows = 18000
last_m_test_rows =  np.shape(data_joke_scores)[0] - first_n_train_rows

"train"
for i in range(first_n_train_rows):
    x = data_joke_scores[i, 0:d]
    for a in range(num_arms):
        theta = np.linalg.solve(A[a], b[a])
        
        cb = alpha * np.sqrt(np.dot(x, np.linalg.solve(A[a], x)))
        ucbs[a] = np.dot(theta, x) + cb
    
    chosen_arm = np.argmax(ucbs)
    A[chosen_arm, :, :] += np.outer(x, x)
    b[chosen_arm, :] += data_joke_scores[i, d+chosen_arm]*x
    
    print(i)    

"test"
test_rewards = []
test_opt = []
for i in range(first_n_train_rows, first_n_train_rows+last_m_test_rows):
    x = data_joke_scores[i, 0:d]
    for a in range(num_arms):
        theta = np.linalg.solve(A[a], b[a])
        ucbs[a] = np.dot(theta, x)
    
    chosen_arm = np.argmax(ucbs)
    test_rewards.append(data_joke_scores[i, d+chosen_arm])
    test_opt.append(data_opt_rewards[i])  

test_regret = np.arange(1,last_m_test_rows+1)*np.mean(test_opt) - np.cumsum(test_rewards)
            
plt.figure(figsize = (8, 6))
plt.plot(test_regret, '.')
plt.xlabel('t', fontsize = 20)
plt.ylabel('regret', fontsize = 20)
plt.savefig('test-regret.png')
plt.close()
