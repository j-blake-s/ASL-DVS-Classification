import matplotlib.pyplot as plt

# Data points from your results
models = ['CNN', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'SNN']
accuracy = [66.25, 72.5, 60.0, 56.25, 51.25, 42.5, 33.75, 32.5, 23.75]

# Color Palette Definitions
# Orange for CNN, Gray for Hybrids, Blue for SNN
# colors = ['#b45cc0'] + ['#00796B']*7 + ['#4f5ebb']
colors = ['#7E1832'] + ['#327E18']*7 + ['#18327E']
# colors = ['#808000'] + ['#228B22']*7 + ['#636B2F']

def create_accuracy_plot():
    # Set the general style to be clean and high-contrast
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Create the Bar Chart
    bars = ax.bar(models, accuracy, color=colors, edgecolor='none', width=0.75)

    # Labeling and Titles
    ax.set_xlabel('Models', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Comparison of Accuracy for Baseline CNN, \nHybrid, and SNN Models', fontsize=16, fontweight='bold', loc='center', pad=20)
    
    # Set Y-axis limits with room for the text labels
    ax.set_ylim(0, 85)

    # Remove top and right "spines" for a modern scientific look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add light horizontal gridlines to help the eye follow the data
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, which='major')
    ax.set_axisbelow(True) # Put grid behind the bars
    # ax.grid(False)
        


    # Add numerical labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

    bars = ax.bar(
        models, 
        accuracy, 
        color=colors, 
        edgecolor='black',  # Or a darker version of the bar color
        linewidth=1.2       # Professional weight for thin, crisp lines
    )
    # Styling individual tick labels for emphasis
    # CNN (Orange)
    # ax.get_xticklabels()[0].set_fontweight('bold')
    # ax.get_xticklabels()[0].set_color(colors[0])
    
    # SNN (Blue)
    # ax.get_xticklabels()[-1].set_fontweight('bold')
    # ax.get_xticklabels()[-1].set_color(colors[-1])

    plt.tight_layout()
    
    # Recommendation: Save as PDF for your LaTeX paper to maintain vector quality
    # plt.savefig('temporal_asl_results.pdf', bbox_inches='tight')
    
    plt.savefig("figs/accuracy_plot.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    create_accuracy_plot()