import matplotlib.pyplot as plt

def plot_metrics(train, test, title, ylabel):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.plot(train, c='b', label='train')
    plt.plot(test, c='r', label='test')

    plt.ylabel(ylabel)
    plt.xlabel('100 Epoch')
    plt.legend()
    plt.show()
