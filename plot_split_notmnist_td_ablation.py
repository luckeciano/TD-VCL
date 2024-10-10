import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
one_step = pd.read_csv('/users/lucelo/vcl-nstepkl/split_notmnist_td_ablation_final_1.csv')
one_step = one_step[one_step['Method'].isin(['λ = 0.1', 'λ = 0.5', 'λ = 0.8', 'λ = 0.99'])]
two_step = pd.read_csv('/users/lucelo/vcl-nstepkl/split_notmnist_td_ablation_final_2.csv')
two_step = two_step[two_step['Method'].isin(['λ = 0.1', 'λ = 0.5', 'λ = 0.8', 'λ = 0.99'])]
three_step = pd.read_csv('/users/lucelo/vcl-nstepkl/split_notmnist_td_ablation_final_3.csv')
three_step = three_step[three_step['Method'].isin(['λ = 0.1', 'λ = 0.5', 'λ = 0.8', 'λ = 0.99'])]
four_step = pd.read_csv('/users/lucelo/vcl-nstepkl/split_notmnist_td_ablation_final_4.csv')
four_step = four_step[four_step['Method'].isin(['λ = 0.1', 'λ = 0.5', 'λ = 0.8', 'λ = 0.99'])]
five_step = pd.read_csv('/users/lucelo/vcl-nstepkl/split_notmnist_td_ablation_final_5.csv')
five_step = five_step[five_step['Method'].isin(['λ = 0.1', 'λ = 0.5', 'λ = 0.8', 'λ = 0.99'])]

fig, axs = plt.subplots(1, 5, figsize=(10 * 5,  6 * 1))

sns.pointplot(x='# tasks', y='Accuracy', data=one_step, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=axs[0])
sns.pointplot(x='# tasks', y='Accuracy', data=two_step, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=axs[1])
sns.pointplot(x='# tasks', y='Accuracy', data=three_step, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=axs[2])
sns.pointplot(x='# tasks', y='Accuracy', data=four_step, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=axs[3])
sns.pointplot(x='# tasks', y='Accuracy', data=five_step, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=axs[4])

plt.savefig('./split_notmnist_nstep_ablation_test.png', bbox_inches='tight')