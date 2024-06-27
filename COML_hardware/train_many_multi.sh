#!/bin/bash
#!/bin/bash
#SBATCH --job-name=train_pnorm_multi        # Job name
#SBATCH --nodes=N                           # Number of nodes
#SBATCH --ntasks=N                          # Number of tasks (processes)
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH -o %j.log                           # Standard output and error log

meta_epochs=1000

for seed in 0
do
    for M in 50
    do
        for p in 2.0
        do
            for p_freq in 2000
            do
                for reg_P in 1e-1
                    do
                    for reg_k_R in 0
                    do
                        for k_R_scale in 1
                        do
                            for k_R_z in 1.4
                            do
                                echo -e "\n============================="
                                echo -e "seed = $seed\nM = $M\npnorm_init = $p\np_freq = $p_freq\nmeta_epochs = 1000"
                                echo -e "reg_P = $reg_P\nreg_k_R = $reg_k_R\nk_R_scale = $k_R_scale\nk_R_z = $k_R_z"
                                echo -e "output_dir = reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}_k_R_z_${k_R_z}_z_training"
                                echo -e "=============================\n"
                                srun --ntasks=1 ./train_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs $meta_epochs --reg_P $reg_P --reg_k_R $reg_k_R --k_R_scale $k_R_scale --k_R_z $k_R_z --output_dir "reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}_k_R_z_${k_R_z}_z_training"
                            done
                        done
                    done
                done
            done
        done
    done
done