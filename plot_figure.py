import matplotlib.pyplot as plt

# Provided data
models = ['Llama 2-7B', 'PMC-Llama-7B', 'Meditron-7B', 'Meditron-70B', 
          'ChatGPT', 'MedRAG-ChatGPT', 'RAG-7B', 'RAG-13B', 'JLMR-7B', 'JLMR-13B']
accuracy = [50.3, 50.85, 53.2, 68.95, 54.9, 59, 62.5, 67.7, 63.5, 70.5]
gpu_time = [36, 446, 620, 42630, 0, 0, 84, 128, 100, 148]

# Creating the scatter plot
plt.figure(figsize=(14, 10))  # Increase figure size to accommodate annotations
scatter = plt.scatter(gpu_time, accuracy, color='blue')

# Setting the x-axis to symlog scale due to wide range of values
plt.xscale('symlog')

# Adding title and labels with font size adjustments
plt.title('Model Performance on Medical QA Tasks', fontsize=30)
plt.xlabel('Training Time on Medical Resources (in GPU hours, symlog scale)', fontsize=30)
plt.ylabel('Accuracy on Medical QA Tasks(%)', fontsize=30)
plt.xlim(left=-5, right=1000000)  # Extend x-axis to accommodate text on left
plt.ylim(bottom=49, top=72)  # Extend y-axis to accommodate text on top
# Annotating the points with model names and adjusting positions for clarity
for i, txt in enumerate(models):
    offset = (10, 10)  # Default offset
    if 'JLMR' in txt:
        offset = (0, 15)  # Slight lift for JLMR models to highlight them
        color = 'red'
        weight = 'bold'
    elif txt == 'Llama 2-7B' or txt == 'RAG-7B':
        offset = (0, -25)  # Lower annotation for Llama 2-7B and RAG-7B
        color = 'black'
        weight = 'normal'
    else:
        color = 'black'
        weight = 'normal'
    # Annotate with adjusted offset and color
    plt.annotate(txt, (gpu_time[i], accuracy[i]), textcoords="offset points", 
                 xytext=offset, ha='center', fontsize=30, color=color, weight=weight)

# Adjusting the plot grid and ticks
plt.grid(True, which="both", ls="--")
plt.tick_params(axis='both', which='major', labelsize=30)

# Tight layout to prevent clipping of tick-labels
plt.tight_layout()

# Displaying the plot
plt.show()

# Saving the plot
plt.savefig('tradeoff.png', bbox_inches='tight')  # Ensure all annotations are within the saved image
