const questions = [
    {
      question: "What is Machine Learning?",
      options: ["A branch of artificial intelligence", "A technique for computers to learn from data", "Both of the above", "None of the above"],
      answer: "Both of the above"
    },
    {
      question: "What are the types of Machine Learning?",
      options: ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "All of the above"],
      answer: "All of the above"
    },
    {
      question: "What is supervised learning?",
      options: ["Learning with labeled data", "Learning with unlabeled data", "Both of the above", "None of the above"],
      answer: "Learning with labeled data"
    },
    {
      question: "What is unsupervised learning?",
      options: ["Learning with labeled data", "Learning with unlabeled data", "Both of the above", "None of the above"],
      answer: "Learning with unlabeled data"
    },
    {
      question: "What is reinforcement learning?",
      options: ["Learning with rewards and punishments", "Learning with labeled data", "Both of the above", "None of the above"],
      answer: "Learning with rewards and punishments"
    },
    {
      question: "What is the objective of a machine learning algorithm?",
      options: ["To minimize error", "To maximize accuracy", "To optimize a specific objective function", "All of the above"],
      answer: "All of the above"
    },
    {
      question: "What is the bias-variance tradeoff?",
      options: ["The tradeoff between bias and variance", "The tradeoff between model complexity and error", "The tradeoff between overfitting and underfitting", "All of the above"],
      answer: "All of the above"
    },
    {
      question: "What is overfitting?",
      options: ["When a model learns the noise in the training data", "When a model performs well on unseen data", "Both of the above", "None of the above"],
      answer: "When a model learns the noise in the training data"
    },
    {
      question: "What is underfitting?",
      options: ["When a model learns the noise in the training data", "When a model performs well on unseen data", "Both of the above", "None of the above"],
      answer: "When a model performs well on unseen data"
    },
    {
      question: "What is cross-validation?",
      options: ["A technique for evaluating machine learning models", "A technique for optimizing hyperparameters", "Both of the above", "None of the above"],
      answer: "Both of the above"
    },
    {
      question: "What are hyperparameters?",
      options: ["Parameters of the model that are learned during training", "Parameters of the model that are set before training", "Both of the above", "None of the above"],
      answer: "Parameters of the model that are set before training"
    },
    {
      question: "What is feature engineering?",
      options: ["The process of selecting the most relevant features for a model", "The process of transforming raw data into meaningful features", "Both of the above", "None of the above"],
      answer: "Both of the above"
    },
    {
      question: "What is dimensionality reduction?",
      options: ["The process of reducing the number of features in a dataset", "The process of increasing the number of features in a dataset", "Both of the above", "None of the above"],
      answer: "The process of reducing the number of features in a dataset"
    },
    {
      question: "What is regularization?",
      options: ["A technique for preventing overfitting in machine learning models", "A technique for increasing the complexity of a model", "Both of the above", "None of the above"],
      answer: "A technique for preventing overfitting in machine learning models"
    },
    {
      question: "What is a neural network?",
      options: ["A type of machine learning model inspired by the human brain", "A type of machine learning model inspired by genetic algorithms", "Both of the above", "None of the above"],
      answer: "A type of machine learning model inspired by the human brain"
    },
    {
      question: "What is a convolutional neural network (CNN)?",
      options: ["A type of neural network designed for image recognition", "A type of neural network designed for natural language processing", "Both of the above", "None of the above"],
      answer: "A type of neural network designed for image recognition"
    },
    {
      question: "What is a recurrent neural network (RNN)?",
      options: ["A type of neural network designed for sequential data", "A type of neural network designed for image recognition", "Both of the above", "None of the above"],
      answer: "A type of neural network designed for sequential data"
    },
    {
      question: "What is deep learning?",
      options: ["A subset of machine learning based on artificial neural networks", "A subset of machine learning based on decision trees", "Both of the above", "None of the above"],
      answer: "A subset of machine learning based on artificial neural networks"
    },
    {
      question: "What is natural language processing (NLP)?",
      options: ["A field of study focused on the interaction between computers and human languages", "A field of study focused on speech recognition", "Both of the above", "None of the above"],
      answer: "A field of study focused on the interaction between computers and human languages"
    },
    {
      question: "What is computer vision?",
      options: ["A field of study focused on teaching computers to interpret visual information", "A field of study focused on teaching computers to understand human language", "Both of the above", "None of the above"],
      answer: "A field of study focused on teaching computers to interpret visual information"
    },
    {
      question: "What is a support vector machine (SVM)?",
      options: ["A type of supervised learning algorithm used for classification and regression tasks", "A type of unsupervised learning algorithm used for clustering tasks", "Both of the above", "None of the above"],
      answer: "A type of supervised learning algorithm used for classification and regression tasks"
    },
    {
      question: "What is a decision tree?",
      options: ["A type of supervised learning algorithm used for classification and regression tasks", "A type of unsupervised learning algorithm used for clustering tasks", "Both of the above", "None of the above"],
      answer: "A type of supervised learning algorithm used for classification and regression tasks"
    },
    {
      question: "What is ensemble learning?",
      options: ["A machine learning technique that combines multiple models to improve performance", "A machine learning technique that focuses on training a single powerful model", "Both of the above", "None of the above"],
      answer: "A machine learning technique that combines multiple models to improve performance"
    },
    {
      question: "What is a random forest?",
      options: ["An ensemble learning technique based on decision trees", "A type of neural network", "Both of the above", "None of the above"],
      answer: "An ensemble learning technique based on decision trees"
    },
    {
      question: "What is clustering?",
      options: ["A type of unsupervised learning algorithm used to group similar data points together", "A type of supervised learning algorithm used for classification tasks", "Both of the above", "None of the above"],
      answer: "A type of unsupervised learning algorithm used to group similar data points together"
    },
    {
      question: "What is regression?",
      options: ["A type of supervised learning algorithm used for predicting continuous values", "A type of unsupervised learning algorithm used for grouping similar data points together", "Both of the above", "None of the above"],
      answer: "A type of supervised learning algorithm used for predicting continuous values"
    },
    {
      question: "What is anomaly detection?",
      options: ["A type of unsupervised learning algorithm used to identify unusual patterns in data", "A type of supervised learning algorithm used for classification tasks", "Both of the above", "None of the above"],
      answer: "A type of unsupervised learning algorithm used to identify unusual patterns in data"
    },
    {
      question: "What is batch normalization?",
      options: ["A technique for improving the performance and stability of neural networks", "A technique for reducing the dimensionality of data", "Both of the above", "None of the above"],
      answer: "A technique for improving the performance and stability of neural networks"
    },
    {
      question: "What is transfer learning?",
      options: ["A machine learning technique where a model trained on one task is applied to a different but related task", "A machine learning technique where a model is trained from scratch on a new task", "Both of the above", "None of the above"],
      answer: "A machine learning technique where a model trained on one task is applied to a different but related task"
    },
    {
      question: "What is reinforcement learning?",
      options: ["A type of machine learning where an agent learns to take actions in an environment to maximize some notion of cumulative reward", "A type of machine learning where an agent learns to classify data into predefined categories", "Both of the above", "None of the above"],
      answer: "A type of machine learning where an agent learns to take actions in an environment to maximize some notion of cumulative reward"
    },
    {
      question: "What is the reward function in reinforcement learning?",
      options: ["A function that assigns a numerical value to each state-action pair", "A function that determines the probability of transitioning from one state to another", "Both of the above", "None of the above"],
      answer: "A function that assigns a numerical value to each state-action pair"
    },
    {
      question: "What is policy in reinforcement learning?",
      options: ["A strategy that determines the agent's behavior in an environment", "A function that assigns a numerical value to each state-action pair", "Both of the above", "None of the above"],
      answer: "A strategy that determines the agent's behavior in an environment"
    },
    {
      question: "What is exploration in reinforcement learning?",
      options: ["The process of trying out different actions to learn more about the environment", "The process of exploiting the current knowledge to maximize rewards", "Both of the above", "None of the above"],
      answer: "The process of trying out different actions to learn more about the environment"
    },
    {
      question: "What is exploitation in reinforcement learning?",
      options: ["The process of exploiting the current knowledge to maximize rewards", "The process of trying out different actions to learn more about the environment", "Both of the above", "None of the above"],
      answer: "The process of exploiting the current knowledge to maximize rewards"
    },
    {
      question: "What is the Bellman equation?",
      options: ["An equation that describes the relationship between the value function of a state and the value function of its successor states", "An equation that describes the relationship between the policy function of a state and the policy function of its successor states", "Both of the above", "None of the above"],
      answer: "An equation that describes the relationship between the value function of a state and the value function of its successor states"
    },
    {
      question: "What is the value function in reinforcement learning?",
      options: ["A function that assigns a value to each state in the environment", "A function that assigns a value to each action in the environment", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each state in the environment"
    },
    {
      question: "What is the Q-function in reinforcement learning?",
      options: ["A function that assigns a value to each state-action pair in the environment", "A function that assigns a value to each state in the environment", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each state-action pair in the environment"
    },
    {
      question: "What is the policy gradient method?",
      options: ["A method for updating the policy of a reinforcement learning agent based on the gradient of the expected reward with respect to the policy parameters", "A method for updating the value function of a reinforcement learning agent based on the gradient of the expected reward with respect to the value function parameters", "Both of the above", "None of the above"],
      answer: "A method for updating the policy of a reinforcement learning agent based on the gradient of the expected reward with respect to the policy parameters"
    },
    {
      question: "What is a Markov decision process (MDP)?",
      options: ["A mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker", "A mathematical framework used to model decision-making in situations where outcomes are fully determined by the actions of a decision-maker", "Both of the above", "None of the above"],
      answer: "A mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker"
    },
    {
      question: "What is the discount factor in reinforcement learning?",
      options: ["A parameter that determines the importance of future rewards in the agent's decision-making process", "A parameter that determines the rate at which rewards are discounted over time", "Both of the above", "None of the above"],
      answer: "A parameter that determines the importance of future rewards in the agent's decision-making process"
    },
    {
      question: "What is policy iteration in reinforcement learning?",
      options: ["An iterative algorithm for finding an optimal policy in a Markov decision process", "An iterative algorithm for updating the policy of a reinforcement learning agent", "Both of the above", "None of the above"],
      answer: "An iterative algorithm for finding an optimal policy in a Markov decision process"
    },
    {
      question: "What is value iteration in reinforcement learning?",
      options: ["An iterative algorithm for finding an optimal value function in a Markov decision process", "An iterative algorithm for updating the value function of a reinforcement learning agent", "Both of the above", "None of the above"],
      answer: "An iterative algorithm for finding an optimal value function in a Markov decision process"
    },
    {
      question: "What is a Markov chain?",
      options: ["A mathematical model used to describe a sequence of events where the probability of each event depends only on the state attained in the previous event", "A mathematical model used to describe a sequence of events where the probability of each event depends on the current state", "Both of the above", "None of the above"],
      answer: "A mathematical model used to describe a sequence of events where the probability of each event depends only on the state attained in the previous event"
    },
    {
      question: "What is the stationary distribution of a Markov chain?",
      options: ["The long-term probability distribution of states in a Markov chain", "The probability distribution of states at a given time step in a Markov chain", "Both of the above", "None of the above"],
      answer: "The long-term probability distribution of states in a Markov chain"
    },
    {
      question: "What is the transition matrix of a Markov chain?",
      options: ["A matrix that describes the probabilities of transitioning between states in a Markov chain", "A matrix that describes the probabilities of states at a given time step in a Markov chain", "Both of the above", "None of the above"],
      answer: "A matrix that describes the probabilities of transitioning between states in a Markov chain"
    },
    {
      question: "What is the absorbing state of a Markov chain?",
      options: ["A state from which there are no transitions to other states", "A state that can transition to other states", "Both of the above", "None of the above"],
      answer: "A state from which there are no transitions to other states"
    },
    {
      question: "What is the ergodicity of a Markov chain?",
      options: ["The property that every state is reachable from every other state", "The property that the Markov chain has a unique stationary distribution", "Both of the above", "None of the above"],
      answer: "The property that every state is reachable from every other state"
    },
    {
      question: "What is the Markov property?",
      options: ["The property that the future state of a system depends only on its current state, not on its past states", "The property that the past states of a system influence its future state", "Both of the above", "None of the above"],
      answer: "The property that the future state of a system depends only on its current state, not on its past states"
    },
    {
      question: "What is the Bellman equation for a Markov decision process?",
      options: ["An equation that describes the relationship between the value function of a state and the value function of its successor states", "An equation that describes the relationship between the policy function of a state and the policy function of its successor states", "Both of the above", "None of the above"],
      answer: "An equation that describes the relationship between the value function of a state and the value function of its successor states"
    },
    {
      question: "What is a partially observable Markov decision process (POMDP)?",
      options: ["A mathematical model used to describe decision-making in situations where the state of the system is only partially observable", "A mathematical model used to describe decision-making in situations where the state of the system is fully observable", "Both of the above", "None of the above"],
      answer: "A mathematical model used to describe decision-making in situations where the state of the system is only partially observable"
    },
    {
      question: "What is the belief state in a partially observable Markov decision process (POMDP)?",
      options: ["A probability distribution over the possible states of the system", "The true state of the system", "Both of the above", "None of the above"],
      answer: "A probability distribution over the possible states of the system"
    },
    {
      question: "What is the observation function in a partially observable Markov decision process (POMDP)?",
      options: ["A function that maps states to observations", "A function that maps observations to states", "Both of the above", "None of the above"],
      answer: "A function that maps states to observations"
    },
    {
      question: "What is the belief update rule in a partially observable Markov decision process (POMDP)?",
      options: ["A rule for updating the belief state based on observations and actions", "A rule for updating the true state of the system based on observations and actions", "Both of the above", "None of the above"],
      answer: "A rule for updating the belief state based on observations and actions"
    },
    {
      question: "What is the history in a partially observable Markov decision process (POMDP)?",
      options: ["A sequence of observations and actions", "A sequence of states and actions", "Both of the above", "None of the above"],
      answer: "A sequence of observations and actions"
    },
    {
      question: "What is the optimal policy in a partially observable Markov decision process (POMDP)?",
      options: ["A policy that maximizes the expected cumulative reward over time", "A policy that maximizes the probability of reaching a goal state", "Both of the above", "None of the above"],
      answer: "A policy that maximizes the expected cumulative reward over time"
    },
    {
      question: "What is the value function in a partially observable Markov decision process (POMDP)?",
      options: ["A function that assigns a value to each belief state", "A function that assigns a value to each action", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each belief state"
    },
    {
      question: "What is the POMDP planner?",
      options: ["An algorithm for finding an optimal policy in a partially observable Markov decision process (POMDP)", "An algorithm for finding an optimal policy in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal policy in a partially observable Markov decision process (POMDP)"
    },
    {
      question: "What is the Monte Carlo Tree Search (MCTS)?",
      options: ["A search algorithm used in decision trees", "A search algorithm used in game tree search", "Both of the above", "None of the above"],
      answer: "A search algorithm used in game tree search"
    },
    {
      question: "What is the value iteration algorithm?",
      options: ["An algorithm for finding an optimal value function in a Markov decision process (MDP)", "An algorithm for finding an optimal policy in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal value function in a Markov decision process (MDP)"
    },
    {
      question: "What is the policy iteration algorithm?",
      options: ["An algorithm for finding an optimal policy in a Markov decision process (MDP)", "An algorithm for finding an optimal value function in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal policy in a Markov decision process (MDP)"
    },
    {
      question: "What is the difference between value iteration and policy iteration?",
      options: ["Value iteration directly computes the optimal value function, while policy iteration alternates between policy evaluation and policy improvement", "Policy iteration directly computes the optimal value function, while value iteration alternates between policy evaluation and policy improvement", "Both of the above", "None of the above"],
      answer: "Value iteration directly computes the optimal value function, while policy iteration alternates between policy evaluation and policy improvement"
    },
    {
      question: "What is reinforcement learning?",
      options: ["A type of machine learning where an agent learns to take actions in an environment to maximize some notion of cumulative reward", "A type of machine learning where an agent learns to classify data into predefined categories", "Both of the above", "None of the above"],
      answer: "A type of machine learning where an agent learns to take actions in an environment to maximize some notion of cumulative reward"
    },
    {
      question: "What is the reward function in reinforcement learning?",
      options: ["A function that assigns a numerical value to each state-action pair", "A function that determines the probability of transitioning from one state to another", "Both of the above", "None of the above"],
      answer: "A function that assigns a numerical value to each state-action pair"
    },
    {
      question: "What is policy in reinforcement learning?",
      options: ["A strategy that determines the agent's behavior in an environment", "A function that assigns a numerical value to each state-action pair", "Both of the above", "None of the above"],
      answer: "A strategy that determines the agent's behavior in an environment"
    },
    {
      question: "What is exploration in reinforcement learning?",
      options: ["The process of trying out different actions to learn more about the environment", "The process of exploiting the current knowledge to maximize rewards", "Both of the above", "None of the above"],
      answer: "The process of trying out different actions to learn more about the environment"
    },
    {
      question: "What is exploitation in reinforcement learning?",
      options: ["The process of exploiting the current knowledge to maximize rewards", "The process of trying out different actions to learn more about the environment", "Both of the above", "None of the above"],
      answer: "The process of exploiting the current knowledge to maximize rewards"
    },
    {
      question: "What is the Bellman equation?",
      options: ["An equation that describes the relationship between the value function of a state and the value function of its successor states", "An equation that describes the relationship between the policy function of a state and the policy function of its successor states", "Both of the above", "None of the above"],
      answer: "An equation that describes the relationship between the value function of a state and the value function of its successor states"
    },
    {
      question: "What is the value function in reinforcement learning?",
      options: ["A function that assigns a value to each state in the environment", "A function that assigns a value to each action in the environment", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each state in the environment"
    },
    {
      question: "What is the Q-function in reinforcement learning?",
      options: ["A function that assigns a value to each state-action pair in the environment", "A function that assigns a value to each state in the environment", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each state-action pair in the environment"
    },
    {
      question: "What is the policy gradient method?",
      options: ["A method for updating the policy of a reinforcement learning agent based on the gradient of the expected reward with respect to the policy parameters", "A method for updating the value function of a reinforcement learning agent based on the gradient of the expected reward with respect to the value function parameters", "Both of the above", "None of the above"],
      answer: "A method for updating the policy of a reinforcement learning agent based on the gradient of the expected reward with respect to the policy parameters"
    },
    {
      question: "What is a Markov decision process (MDP)?",
      options: ["A mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker", "A mathematical framework used to model decision-making in situations where outcomes are fully determined by the actions of a decision-maker", "Both of the above", "None of the above"],
      answer: "A mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker"
    },
    {
      question: "What is the discount factor in reinforcement learning?",
      options: ["A parameter that determines the importance of future rewards in the agent's decision-making process", "A parameter that determines the rate at which rewards are discounted over time", "Both of the above", "None of the above"],
      answer: "A parameter that determines the importance of future rewards in the agent's decision-making process"
    },
    {
      question: "What is policy iteration in reinforcement learning?",
      options: ["An iterative algorithm for finding an optimal policy in a Markov decision process", "An iterative algorithm for updating the policy of a reinforcement learning agent", "Both of the above", "None of the above"],
      answer: "An iterative algorithm for finding an optimal policy in a Markov decision process"
    },
    {
      question: "What is value iteration in reinforcement learning?",
      options: ["An iterative algorithm for finding an optimal value function in a Markov decision process", "An iterative algorithm for updating the value function of a reinforcement learning agent", "Both of the above", "None of the above"],
      answer: "An iterative algorithm for finding an optimal value function in a Markov decision process"
    },
    {
      question: "What is a Markov chain?",
      options: ["A mathematical model used to describe a sequence of events where the probability of each event depends only on the state attained in the previous event", "A mathematical model used to describe a sequence of events where the probability of each event depends on the current state", "Both of the above", "None of the above"],
      answer: "A mathematical model used to describe a sequence of events where the probability of each event depends only on the state attained in the previous event"
    },
    {
      question: "What is the stationary distribution of a Markov chain?",
      options: ["The long-term probability distribution of states in a Markov chain", "The probability distribution of states at a given time step in a Markov chain", "Both of the above", "None of the above"],
      answer: "The long-term probability distribution of states in a Markov chain"
    },
    {
      question: "What is the transition matrix of a Markov chain?",
      options: ["A matrix that describes the probabilities of transitioning between states in a Markov chain", "A matrix that describes the probabilities of states at a given time step in a Markov chain", "Both of the above", "None of the above"],
      answer: "A matrix that describes the probabilities of transitioning between states in a Markov chain"
    },
    {
      question: "What is the absorbing state of a Markov chain?",
      options: ["A state from which there are no transitions to other states", "A state that can transition to other states", "Both of the above", "None of the above"],
      answer: "A state from which there are no transitions to other states"
    },
    {
      question: "What is the ergodicity of a Markov chain?",
      options: ["The property that every state is reachable from every other state", "The property that the Markov chain has a unique stationary distribution", "Both of the above", "None of the above"],
      answer: "The property that every state is reachable from every other state"
    },
    {
      question: "What is the Markov property?",
      options: ["The property that the future state of a system depends only on its current state, not on its past states", "The property that the past states of a system influence its future state", "Both of the above", "None of the above"],
      answer: "The property that the future state of a system depends only on its current state, not on its past states"
    },
    {
      question: "What is the Bellman equation for a Markov decision process?",
      options: ["An equation that describes the relationship between the value function of a state and the value function of its successor states", "An equation that describes the relationship between the policy function of a state and the policy function of its successor states", "Both of the above", "None of the above"],
      answer: "An equation that describes the relationship between the value function of a state and the value function of its successor states"
    },
    {
      question: "What is a partially observable Markov decision process (POMDP)?",
      options: ["A mathematical model used to describe decision-making in situations where the state of the system is only partially observable", "A mathematical model used to describe decision-making in situations where the state of the system is fully observable", "Both of the above", "None of the above"],
      answer: "A mathematical model used to describe decision-making in situations where the state of the system is only partially observable"
    },
    {
      question: "What is the belief state in a partially observable Markov decision process (POMDP)?",
      options: ["A probability distribution over the possible states of the system", "The true state of the system", "Both of the above", "None of the above"],
      answer: "A probability distribution over the possible states of the system"
    },
    {
      question: "What is the observation function in a partially observable Markov decision process (POMDP)?",
      options: ["A function that maps states to observations", "A function that maps observations to states", "Both of the above", "None of the above"],
      answer: "A function that maps states to observations"
    },
    {
      question: "What is the belief update rule in a partially observable Markov decision process (POMDP)?",
      options: ["A rule for updating the belief state based on observations and actions", "A rule for updating the true state of the system based on observations and actions", "Both of the above", "None of the above"],
      answer: "A rule for updating the belief state based on observations and actions"
    },
    {
      question: "What is the history in a partially observable Markov decision process (POMDP)?",
      options: ["A sequence of observations and actions", "A sequence of states and actions", "Both of the above", "None of the above"],
      answer: "A sequence of observations and actions"
    },
    {
      question: "What is the optimal policy in a partially observable Markov decision process (POMDP)?",
      options: ["A policy that maximizes the expected cumulative reward over time", "A policy that maximizes the probability of reaching a goal state", "Both of the above", "None of the above"],
      answer: "A policy that maximizes the expected cumulative reward over time"
    },
    {
      question: "What is the value function in a partially observable Markov decision process (POMDP)?",
      options: ["A function that assigns a value to each belief state", "A function that assigns a value to each action", "Both of the above", "None of the above"],
      answer: "A function that assigns a value to each belief state"
    },
    {
      question: "What is the POMDP planner?",
      options: ["An algorithm for finding an optimal policy in a partially observable Markov decision process (POMDP)", "An algorithm for finding an optimal policy in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal policy in a partially observable Markov decision process (POMDP)"
    },
    {
      question: "What is the Monte Carlo Tree Search (MCTS)?",
      options: ["A search algorithm used in decision trees", "A search algorithm used in game tree search", "Both of the above", "None of the above"],
      answer: "A search algorithm used in game tree search"
    },
    {
      question: "What is the value iteration algorithm?",
      options: ["An algorithm for finding an optimal value function in a Markov decision process (MDP)", "An algorithm for finding an optimal policy in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal value function in a Markov decision process (MDP)"
    },
    {
      question: "What is the policy iteration algorithm?",
      options: ["An algorithm for finding an optimal policy in a Markov decision process (MDP)", "An algorithm for finding an optimal value function in a Markov decision process (MDP)", "Both of the above", "None of the above"],
      answer: "An algorithm for finding an optimal policy in a Markov decision process (MDP)"
    },
    {
      question: "What is the difference between value iteration and policy iteration?",
      options: ["Value iteration directly computes the optimal value function, while policy iteration alternates between policy evaluation and policy improvement", "Policy iteration directly computes the optimal value function, while value iteration alternates between policy evaluation and policy improvement", "Both of the above", "None of the above"],
      answer: "Value iteration directly computes the optimal value function, while policy iteration alternates between policy evaluation and policy improvement"
    }
  ];

let currentQuestionIndex = 0;
let score = 0;

const questionElement = document.getElementById("question");
const optionsContainer = document.getElementById("options-container");
const scoreElement = document.getElementById("score-value");
const restartBtn = document.getElementById("restart-btn");

// Function to display question
function displayQuestion() {
    const currentQuestion = questions[currentQuestionIndex];
    questionElement.textContent = currentQuestion.question;

    optionsContainer.innerHTML = "";
    currentQuestion.options.forEach(option => {
        const optionButton = document.createElement("button");
        optionButton.classList.add("option");
        optionButton.textContent = option;
        optionButton.addEventListener("click", () => checkAnswer(option));
        optionsContainer.appendChild(optionButton);
    });
}

// Function to check answer
function checkAnswer(selectedOption) {
    const currentQuestion = questions[currentQuestionIndex];
    if (selectedOption === currentQuestion.answer) {
        score++;
        currentQuestionIndex++;
        if (currentQuestionIndex < questions.length) {
            displayQuestion();
        } else {
            endGame();
        }
    } else {
        endGame(); // End game if the answer is wrong
    }
}

// Function to end the game
function endGame() {
    questionElement.textContent = "Game Over! Your score is " + score;
    optionsContainer.innerHTML = "";
    restartBtn.style.display = "block";
    scoreElement.textContent = score; // Show the score
    scoreElement.style.display = "block"; // Show the score element
}

// Function to restart the game
restartBtn.addEventListener("click", () => {
    currentQuestionIndex = 0;
    score = 0;
    shuffleQuestions(); // Shuffle questions
    displayQuestion();
    restartBtn.style.display = "none";
    scoreElement.style.display = "none"; // Hide the score
});

// Function to shuffle array elements
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Function to shuffle questions array
function shuffleQuestions() {
    shuffleArray(questions);
    questions.forEach(question => shuffleArray(question.options));
}

// Initial shuffle of questions and options
shuffleQuestions();
displayQuestion();
scoreElement.style.display = "none"; // Hide the score initially
