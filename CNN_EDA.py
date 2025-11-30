import os
import CNN_data_loading
import matplotlib.pyplot as plt

config = CNN_data_loading.config
dataset_train = CNN_data_loading.dataset_train

train_dict = {}
for root, folder, files in os.walk(config["data"]["train_dir_izzy"]):
    for f in folder:
        count = 0
        for path in os.scandir(config["data"]["train_dir_izzy"] + "/" + f):
            if path.is_file():
                count += 1
        train_dict[f] = count

print(train_dict)

#print("Sum of training files using sum of folders: " + str(sum(train_dict.values())) + "\n" + "Total files sanity check: " + str(len(dataset_train)))

keys = list(train_dict.keys())
values = list(train_dict.values())

plt.figure(figsize=(10, 6))
plt.bar(keys, values)
plt.xlabel('emotion', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('training data proportions', fontsize=14)
plt.tight_layout()
plt.show()