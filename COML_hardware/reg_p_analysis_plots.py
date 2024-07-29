import matplotlib as plt
import pickle

if __name__ == "__main__":
    training_file = "./train_results/*.pkl" 

    with open(training_file, "rb") as file:
        data = pickle.read(file)

    train_results = {}

    train_results['A'] = data['train_lossaux_history']['A']
    train_results['reg_p'] = data['regP']
    train_results['validation_loss_history'] = data['valid_loss_history']
    train_results['eig_P'] = data['train_lossaux_history']['eig_P']

    # plt.plot([i for i in range(2500)], test_results['train_info']['valid_loss_history'], marker='o', linestyle='-', color='r')
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Loss History')
    # plt.ylim(0, 5)
    # plt.title('')
    # plt.show()