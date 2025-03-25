#%%
import pandas as pd
import glob

# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-16gens"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou-32samples-piano"
run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou-32samples-piano-random-tempo"

# load all logs

logs = glob.glob(run_path + "/rl_logs/**/*.parquet", recursive=True)

logs = [pd.read_parquet(log) for log in logs]
logs = pd.concat(logs)


# for all columns that are dicts, expand them and prepend dict to the key as name
for col in logs.columns:
    if logs[col].apply(lambda x: isinstance(x, dict)).all():
        logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
        logs.drop(col, axis=1, inplace=True)


# load all midi
midi_paths = glob.glob(run_path + "/midi/**/*.mid", recursive=True)



# create records with 
midi = [{"midi_path": m, "reward_step": int(m.split("/")[-2].split("_")[0]), "idx" : int(m.split("_")[-1].replace(".mid","")) } for m in midi_paths]


#%%
# join logs and midi on reward_step and idx

midi = pd.DataFrame(midi)
logs = logs.merge(midi, on=["reward_step", "idx"], how="inner")

# %%

# print all columns
for col in logs.columns:
    print(col)

# print reward distribution
print(logs["reward"].describe())

# make scatter plot of steps vs reward
import matplotlib.pyplot as plt
plt.scatter(logs["reward_step"], logs["reward"], alpha=0.1)
plt.show()

# plot scatter plot of CE rewards
plt.scatter(logs["reward_step"], logs["normalized_rewards_CE"], alpha=0.1)
plt.show()

# 
# plt.scatter( logs["reward_step"],logs["normalized_rewards_programs_iou"], alpha=0.1, c="green")

# # plot step and content enjoyment
# plt.scatter(logs["reward_step"], logs["normalized_rewards_CE"], alpha=0.1, c="red")

# make violin plot at each step
import seaborn as sns
sns.violinplot(data=logs, x="reward_step", y="normalized_rewards_CE", color="red")

sns.violinplot(data=logs, x="reward_step", y="normalized_rewards_programs_iou", color="green")

# add violin plot for reward
sns.violinplot(data=logs, x="reward_step", y="reward", color="blue")

# plot mean of normalized rewards CE over steps
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].mean(), c="red")
# plot mean of normalized rewards programs iou over steps
plt.plot(logs.groupby("reward_step")["normalized_rewards_programs_iou"].mean(), c="green")
# plot mean of reward over steps
plt.plot(logs.groupby("reward_step")["reward"].mean(), c="blue")
plt.show()




# plot rolling average of rewards (grouped my step) across time
plt.plot(logs.groupby("reward_step")["reward"].mean().rolling(5).mean(), c="blue")
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].mean().rolling(5).mean(), c="red")
plt.show()

# plot the min reward at each step
plt.plot(logs.groupby("reward_step")["reward"].min(), c="blue")
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].min(), c="red")
plt.plot(logs.groupby("reward_step")["normalized_rewards_programs_iou"].min(), c="green")
plt.title("min reward at each step")
plt.show()


#%%

# plot ce, cu, pc, pq mean across steps
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].mean(), c="red", label="CE")
plt.plot(logs.groupby("reward_step")["normalized_rewards_CU"].mean(), c="teal", label="CU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_programs_iou"].mean(), c="green", label="IOU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PC"].mean(), c="orange", label="PC")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PQ"].mean(), c="purple", label="PQ")
# plot average reward in blue
plt.plot(logs.groupby("reward_step")["reward"].mean(), c="blue", label="reward")
plt.legend()
plt.title("Mean reward across steps")
plt.show()


# fit line to reward vs step
import numpy as np
from sklearn.linear_model import LinearRegression
X = logs["reward_step"].values.reshape(-1, 1)
y = logs["normalized_rewards_CE"].values
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, alpha=0.1)
plt.plot(X, reg.predict(X), c="red")
plt.show()









# %%
