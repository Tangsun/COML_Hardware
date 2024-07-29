import pickle

if __name__ == "__main__":
    # with open('train_results/reg_P_1_reg_k_R_0.001_k_R_scale_1_k_R_z_1.26_z_training/seed=0_M=50_E=2500_pinit=2.00_pfreq=2500_regP=1.0000.pkl', 'rb') as file:
    with open('train_results/reg_P_25_reg_k_R_0.001_k_R_scale_1_k_R_z_1.26_z_training/seed=0_M=50_E=2500_pinit=2.00_pfreq=2500_regP=25.0000.pkl', 'rb') as file:
        raw = pickle.load(file)

    # for loss in raw['valid_loss_history']:
    for loss in raw['train_lossaux_history']: 
        for key in loss:
            print(f"{key}:\n{loss[key]}\n")

        # print(loss)

        print("\n\n==========================================================\n\n")