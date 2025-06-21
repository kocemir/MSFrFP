import torch
import matplotlib.pyplot as plt
checkpoint = torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/resnet50/cifar100/frft/2023-11-28_12-37-47_bc_64_ep_38_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")

model=checkpoint["net"]
# Count the total number of parameters
total_params = sum(p.numel() for p in model.values())
print(f"Total Parameters: {total_params}")

print("Epoch",checkpoint['epoch'])
print("Best Epoch", checkpoint['best_epoch'])
print("Train accuracy",checkpoint['train_acc'])
print("Valid accuracy",checkpoint['valid_acc'])
print("Best Valid accuracy",checkpoint['best_valid_acc'])
print("Best Valid Top5 Accuracy", checkpoint['best_valid_top5_acc'])

print("Final Test accuracy",checkpoint['test_acc'])
print("Best Test accuracy",checkpoint['best_test_acc'])
print("Best Test Top5 Accuracy", checkpoint['best_test_top5_acc'])

print("Elapsed time",checkpoint["training_time"])
print("Mean inference time {}".format(checkpoint['mean_inference_time']))
print("Frac a",checkpoint["frac_a"][-1])
print("Frac b",checkpoint["frac_b"][-1])




########### PLOTTING WILL BE HERE #################

### SELECTED SOME MODELS AND DATASET, PLOTTED THEM. IF YOU WISH TO PLOT ALL, YOU SHOULD FOLLOW LOADING ALL

# DenseNet
checkpoint_densenet_cub2011 = torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/densenet121/cub2011/frft/2023-11-29_19-05-01_bc_64_ep_42_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")
checkpoint_densenet_cifar100 = torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/densenet121/cifar100/frft/2023-11-26_01-47-13_bc_64_ep_43_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")


checkpoint_resnet50_cifar100= torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/resnet50/cifar100/frft/2023-11-28_12-37-47_bc_64_ep_38_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")
checkpoint_resnet50_cub2011= torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/resnet50/cub2011/frft/2023-11-28_11-16-21_bc_64_ep_29_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")

checkpoint_vgg13_cub2011= torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/vgg13/cub2011/frft/2023-11-29_12-12-35_bc_32_ep_30_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")
checkpoint_vgg13_cifar100 = torch.load("/auto/k2/aykut3/emirhan_frft/checkpoints/vgg13/cifar100/frft/2023-11-29_20-23-34_bc_32_ep_76_dm_frft_rn_1_lr_0.001_ps_16_pt_10_.pth",map_location="cpu")



########################## DENSENET121 CIFAR100
plt.figure()
plt.plot(checkpoint_densenet_cifar100["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_densenet_cifar100["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("DenseNet121",fontweight='bold')

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/densenet_cifar100')
plt.show()

############################ DENSENET121 CUB2011 
plt.figure()
plt.plot(checkpoint_densenet_cub2011["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_densenet_cub2011["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("DenseNet121",fontweight='bold')

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/densenet_cub2011')
plt.show()

############################ RESNET50 CUB2011 
plt.figure()
plt.plot(checkpoint_resnet50_cub2011["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_resnet50_cub2011["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("ResNet50",fontweight='bold')

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/resnet50_cub2011')
plt.show()

############################ RESNET50 CIFAR100 
plt.figure()
plt.plot(checkpoint_resnet50_cifar100["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_resnet50_cifar100["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("ResNet50",fontweight="bold")

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/resnet50_cifar100')
plt.show()
############################ VGG13 CIFAR100 
plt.figure()
plt.plot(checkpoint_vgg13_cifar100["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_vgg13_cifar100["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("VGG13",fontweight="bold")

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/vgg13_cifar100')
plt.show()

############################ VGG13 CIFAR100 
plt.figure()
plt.plot(checkpoint_vgg13_cub2011["frac_a"], linewidth=2.5,)
plt.plot(checkpoint_vgg13_cub2011["frac_b"], linewidth=2.5)
# Set LaTeX-style axis labels
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Value', fontsize=15,fontweight='bold')
plt.legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
# Make tick labels bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title("VGG13",fontweight='bold')

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine
plt.savefig('./train_figures/vgg13_cub2011')

plt.show()













'''
# Assuming your code for loading checkpoints is defined

# Create subplots in a 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Plot 1: DenseNet CUB2011
axes[0, 0].plot(checkpoint_densenet_cub2011["frac_a"], linewidth=2.5)
axes[0, 0].plot(checkpoint_densenet_cub2011["frac_b"], linewidth=2.5)
axes[0, 0].set_title('DenseNet CUB2011')
axes[0, 0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[0, 0].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')


# Plot 2: DenseNet CIFAR100
axes[0, 1].plot(checkpoint_densenet_cifar100["frac_a"], linewidth=2.5)
axes[0, 1].plot(checkpoint_densenet_cifar100["frac_b"], linewidth=2.5)
axes[0, 1].set_title('DenseNet CIFAR100')
axes[0, 1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[0, 1].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Plot 3: ResNet50 CIFAR100
axes[1, 0].plot(checkpoint_resnet50_cifar100["frac_a"], linewidth=2.5)
axes[1, 0].plot(checkpoint_resnet50_cifar100["frac_b"], linewidth=2.5)
axes[1, 0].set_title('ResNet50 CIFAR100')
axes[1, 0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[1, 0].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Plot 4: ResNet50 CUB2011
axes[1, 1].plot(checkpoint_resnet50_cub2011["frac_a"], linewidth=2.5)
axes[1, 1].plot(checkpoint_resnet50_cub2011["frac_b"], linewidth=2.5)
axes[1, 1].set_title('ResNet50 CUB2011')
axes[1, 1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[1, 1].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Plot 5: VGG13 CUB2011
axes[2, 0].plot(checkpoint_vgg13_cub2011["frac_a"], linewidth=2.5)
axes[2, 0].plot(checkpoint_vgg13_cub2011["frac_b"], linewidth=2.5)
axes[2, 0].set_title('VGG13 CUB2011')
axes[2, 0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[2, 0].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Plot 6: VGG13 CIFAR100
axes[2, 1].plot(checkpoint_vgg13_cifar100["frac_a"], linewidth=2.5)
axes[2, 1].plot(checkpoint_vgg13_cifar100["frac_b"], linewidth=2.5)
axes[2, 1].set_title('VGG13 CIFAR100')
axes[2, 1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
axes[2, 1].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[2, 1].legend([r'$a_{1}$', r'$a_{2}$'], loc='upper right', prop={'weight': 'bold'})
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Get the current Axes object
ax = plt.gca()
# Make the spines (axis frame) bold
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['top'].set_linewidth(2)     # Top spine
ax.spines['left'].set_linewidth(2)    # Left spine
ax.spines['right'].set_linewidth(2)   # Right spine



# Adjust layout
plt.tight_layout()

'''


