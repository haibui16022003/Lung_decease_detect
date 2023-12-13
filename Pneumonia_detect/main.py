from preprocess_data import *
from model import *
import matplotlib.pyplot as plt
# # Visualize example pictures
# fig, axis = plt.subplots(3, 3, figsize=(9, 9))
# c = 0
# for i in range(3):
#     for j in range(3):
#         patient_id = train_labels.patientId.iloc[c]
#         dcm_path = ROOT_PATH / patient_id
#         dcm_path = dcm_path.with_suffix(".dcm")
#         dcm = pydicom.read_file(dcm_path).pixel_array
#
#         label = train_labels["Target"].iloc[c]
#
#         axis[i][j].imshow(dcm, cmap="bone")
#         axis[i][j].set_title(label)
#         c += 1
# plt.show()

new_labels = train_labels.head(6000)
mean, std = preprocess(new_labels)

print(mean, std)

train_dataset, val_dataset = create_dataset(mean, std)

fig, axis = plt.subplots(2, 2, figsize=(9, 9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0, 20000)
        x_ray, label = train_dataset[random_index]
        axis[i][j].imshow(x_ray[0], cmap="bone")
        axis[i][j].set_title(f"Label:{label}")
plt.show()


print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")