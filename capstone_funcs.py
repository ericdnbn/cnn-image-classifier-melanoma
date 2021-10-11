def visualize_filepath(filepath):
	fig, axs = plt.subplots(1, 5, figsize=(20, 5))

	for i, ax in enumerate(axs):
    	ax.imshow(filepath[i], cmap='gray')
	fig.suptitle('Normal Lungs', fontsize=15, y=.8)
	fig.tight_layout;