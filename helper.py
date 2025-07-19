import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training..')
    plt.xlabel('No of games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.001)  # Forces update
    plt.draw()
    
