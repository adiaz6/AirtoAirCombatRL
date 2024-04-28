import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def comparison_plots(r_phase1, r_phase2, r_phase3, success_ph1, success_ph2, success_ph3):
    if not os.path.exists('./test_data/'):
        os.makedirs('./test_data/')

    # Bar Plots
    names = ('Baseline', 'Phase I', 'Phase II')
    num_trials = 10000
    counts = {'Win': 100*np.array([success_ph3['pursuer succeeds'], success_ph1['pursuer succeeds'], success_ph2['pursuer succeeds']]), 
              'Technical win': 100*np.array([success_ph3['evader cornered'], success_ph1['evader cornered'], success_ph2['evader cornered']]), 
              'Failed': 100*np.array([success_ph3['evader succeeds'], success_ph1['evader succeeds'], success_ph2['evader succeeds']])}
    
    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    width=0.6

    for term, count in counts.items():
        p = ax.bar(names, count, width, label=term, bottom=bottom)
        bottom += count

        ax.bar_label(p, label_type='center')

    ax.set_title('Success Rate')
    ax.set_ylim([0, 100])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./test_data/success_rate.png')

    # Plot rewards
     #Pad or truncate episodes to make them of equal length

   # max_length1 = max(len(episode) for episode in r_phase1)
    #padded_rewards1 = [episode[:min_length1] for episode in r_phase1]
    #padded_rewards1 = [np.pad(episode, (0, max_length1 - len(episode)), mode='constant', constant_values=np.nan) for episode in r_phase1]

    # Compute mean and standard deviation across episodes for each time step
    # Compute mean for every time step across all episodes
    mean_rewards1 = np.mean(r_phase1, axis=0)

    #mean_rewards1 = np.mean(padded_rewards1, axis=0)
    std_rewards1 = np.std(r_phase1, axis=0)

    # Calculate confidence intervals
    confidence_interval1 = 1.96 * std_rewards1 / np.sqrt(len(mean_rewards1))

    #min_length2 = min(len(episode) for episode in r_phase2)
    #max_length2 = max(len(episode) for episode in r_phase2)
    #padded_rewards2 = [episode[:min_length2] for episode in r_phase2]
    #padded_rewards2 = [np.pad(episode, (0, max_length2 - len(episode)), mode='constant', constant_values=np.nan) for episode in r_phase2]

    # Compute mean and standard deviation across episodes for each time step
    mean_rewards2 = np.mean(r_phase2, axis=0)
    std_rewards2 = np.std(r_phase2, axis=0)

    # Calculate confidence intervals
    confidence_interval2 = 1.96 * std_rewards2 / np.sqrt(len(mean_rewards2))

    #min_length3 = min(len(episode) for episode in r_phase3)
    #max_length3 = max(len(episode) for episode in r_phase3)
    #padded_rewards3 = [episode[:min_length3] for episode in r_phase3]
    #padded_rewards3 = [np.pad(episode, (0, max_length3 - len(episode)), mode='constant', constant_values=np.nan) for episode in r_phase3]

    # Compute mean and standard deviation across episodes for each time step
    mean_rewards3 = np.mean(r_phase3, axis=0)
    std_rewards3 = np.std(r_phase3, axis=0)

    # Calculate confidence intervals
    confidence_interval3 = 1.96 * std_rewards3 / np.sqrt(len(mean_rewards3))

    # Plot mean rewards with error bars representing confidence intervals
    plt.figure()
    plt.plot(np.arange(len(mean_rewards3)-1), mean_rewards3[:-1], label='Baseline', color='#1f77b4')
    #plt.errorbar(np.arange(len(mean_rewards)), mean_rewards, yerr=confidence_interval, label='Mean Reward', fmt='-o')
    plt.fill_between(np.arange(len(mean_rewards3)-1), mean_rewards3[:-1] - confidence_interval3[:-1], mean_rewards3[:-1]+confidence_interval3[:-1], color='#1f77b4', alpha=0.1)
    plt.plot(np.arange(len(mean_rewards1)-1), mean_rewards1[:-1], label='Phase I', color='#ff7f0e')
    #plt.errorbar(np.arange(len(mean_rewards)), mean_rewards, yerr=confidence_interval, label='Mean Reward', fmt='-o')
    plt.fill_between(np.arange(len(mean_rewards1)-1), mean_rewards1[:-1] - confidence_interval1[:-1], mean_rewards1[:-1]+confidence_interval1[:-1], color='#ff7f0e', alpha=0.1)
    plt.plot(np.arange(len(mean_rewards2)-1), mean_rewards2[:-1], label='Phase II', color='#2ca02c')
    #plt.errorbar(np.arange(len(mean_rewards)), mean_rewards, yerr=confidence_interval, label='Mean Reward', fmt='-o')
    plt.fill_between(np.arange(len(mean_rewards2)-1), mean_rewards2[:-1] - confidence_interval2[:-1], mean_rewards2[:-1]+confidence_interval2[:-1], color='#2ca02c', alpha=0.1)
    plt.xlabel('Time Step')
    plt.ylabel('Accumulation of Reward')
    plt.title('Accumulation of Reward over Episodes')
    plt.legend()
    plt.savefig('./test_data/rewards.png')

def reward_plots(phase):
    # Generate training plots
    rewards = np.load(f'data_{phase}/rewards.npy')
    cumsum_vec = np.cumsum(np.insert(rewards, 0, 0)) 
    ma_vec = (cumsum_vec[100:] - cumsum_vec[:-100]) / 100
    plt.figure()
    plt.plot(range(len(rewards)), rewards, alpha=0.3)
    plt.plot(range(len(ma_vec)), ma_vec, color='#1f77b4', label='Rolling average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Accumulated Reward per Episode')
    plt.legend()
    plt.savefig(f'figures_{phase}/rolling_avg_rewards_{phase}.png')

if __name__ == "__main__":
    phase = sys.argv[1]
    reward_plots(phase)
