## Entry point to run your file pipeline#####

from train import train_model
from evaluate import plot_history

if __name__ == "__main__":
    model, history = train_model()
    plot_history(history)
